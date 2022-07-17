import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
import torch.nn.init as init
from tqdm import tqdm

from models.modules.pseudo_gt import expand_area
from models.model import get_discriminator, get_generator, vgg16
from models.loss import GANLoss, MakeupLoss, ComposePGT, AnnealingComposePGT

from training.utils import plot_curves

class Solver():
    def __init__(self, config, args, logger=None, inference=False):
        self.G = get_generator(config)
        if inference:
            self.G.load_state_dict(torch.load(inference, map_location=args.device))
            self.G = self.G.to(args.device).eval()
            return
        self.double_d = config.TRAINING.DOUBLE_D
        self.D_A = get_discriminator(config)
        if self.double_d:
            self.D_B = get_discriminator(config)
        
        self.load_folder = args.load_folder
        self.save_folder = args.save_folder
        self.vis_folder = os.path.join(args.save_folder, 'visualization')
        if not os.path.exists(self.vis_folder):
            os.makedirs(self.vis_folder)
        self.vis_freq = config.LOG.VIS_FREQ
        self.save_freq = config.LOG.SAVE_FREQ

        # Data & PGT
        self.img_size = config.DATA.IMG_SIZE
        self.margins = {'eye':config.PGT.EYE_MARGIN,
                        'lip':config.PGT.LIP_MARGIN}
        self.pgt_annealing = config.PGT.ANNEALING
        if self.pgt_annealing:
            self.pgt_maker = AnnealingComposePGT(self.margins, 
                config.PGT.SKIN_ALPHA_MILESTONES, config.PGT.SKIN_ALPHA_VALUES,
                config.PGT.EYE_ALPHA_MILESTONES, config.PGT.EYE_ALPHA_VALUES,
                config.PGT.LIP_ALPHA_MILESTONES, config.PGT.LIP_ALPHA_VALUES
            )
        else:
            self.pgt_maker = ComposePGT(self.margins, 
                config.PGT.SKIN_ALPHA,
                config.PGT.EYE_ALPHA,
                config.PGT.LIP_ALPHA
            )
        self.pgt_maker.eval()

        # Hyper-param
        self.num_epochs = config.TRAINING.NUM_EPOCHS
        self.g_lr = config.TRAINING.G_LR
        self.d_lr = config.TRAINING.D_LR
        self.beta1 = config.TRAINING.BETA1
        self.beta2 = config.TRAINING.BETA2
        self.lr_decay_factor = config.TRAINING.LR_DECAY_FACTOR

        # Loss param
        self.lambda_idt      = config.LOSS.LAMBDA_IDT
        self.lambda_A        = config.LOSS.LAMBDA_A
        self.lambda_B        = config.LOSS.LAMBDA_B
        self.lambda_lip  = config.LOSS.LAMBDA_MAKEUP_LIP
        self.lambda_skin = config.LOSS.LAMBDA_MAKEUP_SKIN
        self.lambda_eye  = config.LOSS.LAMBDA_MAKEUP_EYE
        self.lambda_vgg      = config.LOSS.LAMBDA_VGG

        self.device = args.device
        self.keepon = args.keepon
        self.logger = logger
        self.loss_logger = {
            'D-A-loss_real':[],
            'D-A-loss_fake':[],
            'D-B-loss_real':[],
            'D-B-loss_fake':[],
            'G-A-loss-adv':[],
            'G-B-loss-adv':[],
            'G-loss-idt':[],
            'G-loss-img-rec':[],
            'G-loss-vgg-rec':[],
            'G-loss-rec':[],
            'G-loss-skin-pgt':[],
            'G-loss-eye-pgt':[],
            'G-loss-lip-pgt':[],
            'G-loss-pgt':[],
            'G-loss':[],
            'D-A-loss':[],
            'D-B-loss':[]
        }

        self.build_model()
        super(Solver, self).__init__()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        if self.logger is not None:
            self.logger.info('{:s}, the number of parameters: {:d}'.format(name, num_params))
        else:
            print('{:s}, the number of parameters: {:d}'.format(name, num_params))
    
    # For generator
    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)

    def build_model(self):
        self.G.apply(self.weights_init_xavier)
        self.D_A.apply(self.weights_init_xavier)
        if self.double_d:
            self.D_B.apply(self.weights_init_xavier)
        if self.keepon:
            self.load_checkpoint()
        
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(gan_mode='lsgan')
        self.criterionPGT = MakeupLoss()
        self.vgg = vgg16(pretrained=True)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), self.d_lr, [self.beta1, self.beta2])
        if self.double_d:
            self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, 
                    T_max=self.num_epochs, eta_min=self.g_lr * self.lr_decay_factor)
        self.d_A_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_A_optimizer, 
                    T_max=self.num_epochs, eta_min=self.d_lr * self.lr_decay_factor)
        if self.double_d:
            self.d_B_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_B_optimizer, 
                    T_max=self.num_epochs, eta_min=self.d_lr * self.lr_decay_factor)

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D_A, 'D_A')
        if self.double_d: self.print_network(self.D_B, 'D_B')

        self.G.to(self.device)
        self.vgg.to(self.device)
        self.D_A.to(self.device)
        if self.double_d: self.D_B.to(self.device)

    def train(self, data_loader):
        self.len_dataset = len(data_loader)
        
        for self.epoch in range(1, self.num_epochs + 1):
            self.start_time = time.time()
            loss_tmp = self.get_loss_tmp()
            self.G.train(); self.D_A.train(); 
            if self.double_d: self.D_B.train()
            losses_G = []; losses_D_A = []; losses_D_B = []
            
            with tqdm(data_loader, desc="training") as pbar:
                for step, (source, reference) in enumerate(pbar):
                    # image, mask, diff, lms
                    image_s, image_r = source[0].to(self.device), reference[0].to(self.device) # (b, c, h, w)
                    mask_s_full, mask_r_full = source[1].to(self.device), reference[1].to(self.device) # (b, c', h, w) 
                    diff_s, diff_r = source[2].to(self.device), reference[2].to(self.device) # (b, 136, h, w)
                    lms_s, lms_r = source[3].to(self.device), reference[3].to(self.device) # (b, K, 2)

                    # process input mask
                    mask_s = torch.cat((mask_s_full[:,0:1], mask_s_full[:,1:].sum(dim=1, keepdim=True)), dim=1)
                    mask_r = torch.cat((mask_r_full[:,0:1], mask_r_full[:,1:].sum(dim=1, keepdim=True)), dim=1)
                    #mask_s = mask_s_full[:,:2]; mask_r = mask_r_full[:,:2]

                    # ================= Generate ================== #
                    fake_A = self.G(image_s, image_r, mask_s, mask_r, diff_s, diff_r, lms_s, lms_r)
                    fake_B = self.G(image_r, image_s, mask_r, mask_s, diff_r, diff_s, lms_r, lms_s)

                    # generate pseudo ground truth
                    pgt_A = self.pgt_maker(image_s, image_r, mask_s_full, mask_r_full, lms_s, lms_r)
                    pgt_B = self.pgt_maker(image_r, image_s, mask_r_full, mask_s_full, lms_r, lms_s)
                    
                    # ================== Train D ================== #
                    # training D_A, D_A aims to distinguish class B
                    # Real
                    out = self.D_A(image_r)
                    d_loss_real = self.criterionGAN(out, True)
                    # Fake
                    out = self.D_A(fake_A.detach())
                    d_loss_fake =  self.criterionGAN(out, False)

                    # Backward + Optimize
                    d_loss = (d_loss_real + d_loss_fake) * 0.5
                    self.d_A_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_A_optimizer.step()                   

                    # Logging
                    loss_tmp['D-A-loss_real'] += d_loss_real.item()
                    loss_tmp['D-A-loss_fake'] += d_loss_fake.item()
                    losses_D_A.append(d_loss.item())

                    # training D_B, D_B aims to distinguish class A
                    # Real
                    if self.double_d:
                        out = self.D_B(image_s)
                    else:
                        out = self.D_A(image_s)
                    d_loss_real = self.criterionGAN(out, True)
                    # Fake
                    if self.double_d:
                        out = self.D_B(fake_B.detach())
                    else:
                        out = self.D_A(fake_B.detach())
                    d_loss_fake =  self.criterionGAN(out, False)

                    # Backward + Optimize
                    d_loss = (d_loss_real+ d_loss_fake) * 0.5
                    if self.double_d:
                        self.d_B_optimizer.zero_grad()
                        d_loss.backward()
                        self.d_B_optimizer.step()
                    else:
                        self.d_A_optimizer.zero_grad()
                        d_loss.backward()
                        self.d_A_optimizer.step()

                    # Logging
                    loss_tmp['D-B-loss_real'] += d_loss_real.item()
                    loss_tmp['D-B-loss_fake'] += d_loss_fake.item()
                    losses_D_B.append(d_loss.item())

                    # ================== Train G ================== #
                    
                    # G should be identity if ref_B or org_A is fed
                    idt_A = self.G(image_s, image_s, mask_s, mask_s, diff_s, diff_s, lms_s, lms_s)
                    idt_B = self.G(image_r, image_r, mask_r, mask_r, diff_r, diff_r, lms_r, lms_r)
                    loss_idt_A = self.criterionL1(idt_A, image_s) * self.lambda_A * self.lambda_idt
                    loss_idt_B = self.criterionL1(idt_B, image_r) * self.lambda_B * self.lambda_idt
                    # loss_idt
                    loss_idt = (loss_idt_A + loss_idt_B) * 0.5

                    # GAN loss D_A(G_A(A))
                    pred_fake = self.D_A(fake_A)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)

                    # GAN loss D_B(G_B(B))
                    if self.double_d:
                        pred_fake = self.D_B(fake_B)
                    else:
                        pred_fake = self.D_A(fake_B)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)
                    
                    # Makeup loss
                    g_A_loss_pgt = 0; g_B_loss_pgt = 0
                    
                    g_A_lip_loss_pgt = self.criterionPGT(fake_A, pgt_A, mask_s_full[:,0:1]) * self.lambda_lip
                    g_B_lip_loss_pgt = self.criterionPGT(fake_B, pgt_B, mask_r_full[:,0:1]) * self.lambda_lip
                    g_A_loss_pgt += g_A_lip_loss_pgt
                    g_B_loss_pgt += g_B_lip_loss_pgt

                    mask_s_eye = expand_area(mask_s_full[:,2:4].sum(dim=1, keepdim=True), self.margins['eye'])
                    mask_r_eye = expand_area(mask_r_full[:,2:4].sum(dim=1, keepdim=True), self.margins['eye'])
                    mask_s_eye = mask_s_eye * mask_s_full[:,1:2]
                    mask_r_eye = mask_r_eye * mask_r_full[:,1:2]
                    g_A_eye_loss_pgt = self.criterionPGT(fake_A, pgt_A, mask_s_eye) * self.lambda_eye
                    g_B_eye_loss_pgt = self.criterionPGT(fake_B, pgt_B, mask_r_eye) * self.lambda_eye
                    g_A_loss_pgt += g_A_eye_loss_pgt
                    g_B_loss_pgt += g_B_eye_loss_pgt
                    
                    mask_s_skin = mask_s_full[:,1:2] * (1 - mask_s_eye)
                    mask_r_skin = mask_r_full[:,1:2] * (1 - mask_r_eye)
                    g_A_skin_loss_pgt = self.criterionPGT(fake_A, pgt_A, mask_s_skin) * self.lambda_skin
                    g_B_skin_loss_pgt = self.criterionPGT(fake_B, pgt_B, mask_r_skin) * self.lambda_skin
                    g_A_loss_pgt += g_A_skin_loss_pgt
                    g_B_loss_pgt += g_B_skin_loss_pgt
                    
                    # cycle loss
                    rec_A = self.G(fake_A, image_s, mask_s, mask_s, diff_s, diff_s, lms_s, lms_s)
                    rec_B = self.G(fake_B, image_r, mask_r, mask_r, diff_r, diff_r, lms_r, lms_r)

                    # cycle loss v2
                    # rec_A = self.G(fake_A, fake_B, mask_s, mask_r, diff_s, diff_r, lms_s, lms_r)
                    # rec_B = self.G(fake_B, fake_A, mask_r, mask_s, diff_r, diff_s, lms_r, lms_s)

                    g_loss_rec_A = self.criterionL1(rec_A, image_s) * self.lambda_A
                    g_loss_rec_B = self.criterionL1(rec_B, image_r) * self.lambda_B

                    # vgg loss
                    vgg_s = self.vgg(image_s).detach()
                    vgg_fake_A = self.vgg(fake_A)
                    g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_s) * self.lambda_A * self.lambda_vgg

                    vgg_r = self.vgg(image_r).detach()
                    vgg_fake_B = self.vgg(fake_B)
                    g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_r) * self.lambda_B * self.lambda_vgg

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5

                    # Combined loss
                    g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_pgt + g_B_loss_pgt

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss_tmp['G-A-loss-adv'] += g_A_loss_adv.item()
                    loss_tmp['G-B-loss-adv'] += g_B_loss_adv.item()
                    loss_tmp['G-loss-idt'] += loss_idt.item()
                    loss_tmp['G-loss-img-rec'] += (g_loss_rec_A + g_loss_rec_B).item() * 0.5
                    loss_tmp['G-loss-vgg-rec'] += (g_loss_A_vgg + g_loss_B_vgg).item() * 0.5
                    loss_tmp['G-loss-rec'] += loss_rec.item()
                    loss_tmp['G-loss-skin-pgt'] += (g_A_skin_loss_pgt + g_B_skin_loss_pgt).item()
                    loss_tmp['G-loss-eye-pgt'] += (g_A_eye_loss_pgt + g_B_eye_loss_pgt).item()
                    loss_tmp['G-loss-lip-pgt'] += (g_A_lip_loss_pgt + g_B_lip_loss_pgt).item()
                    loss_tmp['G-loss-pgt'] += (g_A_loss_pgt + g_B_loss_pgt).item()
                    losses_G.append(g_loss.item())
                    pbar.set_description("Epoch: %d, Step: %d, Loss_G: %0.4f, Loss_A: %0.4f, Loss_B: %0.4f" % \
                                (self.epoch, step + 1, np.mean(losses_G), np.mean(losses_D_A), np.mean(losses_D_B)))

            self.end_time = time.time()
            for k, v in loss_tmp.items():
                loss_tmp[k] = v / self.len_dataset  
            loss_tmp['G-loss'] = np.mean(losses_G)
            loss_tmp['D-A-loss'] = np.mean(losses_D_A)
            loss_tmp['D-B-loss'] = np.mean(losses_D_B)
            self.log_loss(loss_tmp)
            self.plot_loss()

            # Decay learning rate
            self.g_scheduler.step()
            self.d_A_scheduler.step()
            if self.double_d:
                self.d_B_scheduler.step()

            if self.pgt_annealing:
                self.pgt_maker.step()

            #save the images
            if (self.epoch) % self.vis_freq == 0:
                self.vis_train([image_s.detach().cpu(), 
                                image_r.detach().cpu(), 
                                fake_A.detach().cpu(), 
                                pgt_A.detach().cpu()])
            #                   rec_A.detach().cpu()])

            # Save model checkpoints
            if (self.epoch) % self.save_freq == 0:
                self.save_models()
   

    def get_loss_tmp(self):
        loss_tmp = {
            'D-A-loss_real':0.0,
            'D-A-loss_fake':0.0,
            'D-B-loss_real':0.0,
            'D-B-loss_fake':0.0,
            'G-A-loss-adv':0.0,
            'G-B-loss-adv':0.0,
            'G-loss-idt':0.0,
            'G-loss-img-rec':0.0,
            'G-loss-vgg-rec':0.0,
            'G-loss-rec':0.0,
            'G-loss-skin-pgt':0.0,
            'G-loss-eye-pgt':0.0,
            'G-loss-lip-pgt':0.0,
            'G-loss-pgt':0.0,
        }
        return loss_tmp

    def log_loss(self, loss_tmp):
        if self.logger is not None:
            self.logger.info('\n' + '='*40 + '\nEpoch {:d}, time {:.2f} s'
                            .format(self.epoch, self.end_time - self.start_time))
        else:
            print('\n' + '='*40 + '\nEpoch {:d}, time {:d} s'
                    .format(self.epoch, self.end_time - self.start_time))
        for k, v in loss_tmp.items():
            self.loss_logger[k].append(v)
            if self.logger is not None:
                self.logger.info('{:s}\t{:.6f}'.format(k, v))  
            else:
                print('{:s}\t{:.6f}'.format(k, v))  
        if self.logger is not None:
            self.logger.info('='*40)  
        else:
            print('='*40)

    def plot_loss(self):
        G_losses = []; G_names = []
        D_A_losses = []; D_A_names = []
        D_B_losses = []; D_B_names = []
        D_P_losses = []; D_P_names = []
        for k, v in self.loss_logger.items():
            if 'G' in k:
                G_names.append(k); G_losses.append(v)
            elif 'D-A' in k:
                D_A_names.append(k); D_A_losses.append(v)
            elif 'D-B' in k:
                D_B_names.append(k); D_B_losses.append(v)
            elif 'D-P' in k:
                D_P_names.append(k); D_P_losses.append(v)
        plot_curves(self.save_folder, 'G_loss', G_losses, G_names, ylabel='Loss')
        plot_curves(self.save_folder, 'D-A_loss', D_A_losses, D_A_names, ylabel='Loss')
        plot_curves(self.save_folder, 'D-B_loss', D_B_losses, D_B_names, ylabel='Loss')

    def load_checkpoint(self):
        G_path = os.path.join(self.load_folder, 'G.pth')
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path, map_location=self.device))
            print('loaded trained generator {}..!'.format(G_path))
        D_A_path = os.path.join(self.load_folder, 'D_A.pth')
        if os.path.exists(D_A_path):
            self.D_A.load_state_dict(torch.load(D_A_path, map_location=self.device))
            print('loaded trained discriminator A {}..!'.format(D_A_path))

        if self.double_d:
            D_B_path = os.path.join(self.load_folder, 'D_B.pth')
            if os.path.exists(D_B_path):
                self.D_B.load_state_dict(torch.load(D_B_path, map_location=self.device))
                print('loaded trained discriminator B {}..!'.format(D_B_path))
    
    def save_models(self):
        save_dir = os.path.join(self.save_folder, 'epoch_{:d}'.format(self.epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, 'G.pth'))
        torch.save(self.D_A.state_dict(), os.path.join(save_dir, 'D_A.pth'))
        if self.double_d:
            torch.save(self.D_B.state_dict(), os.path.join(save_dir, 'D_B.pth'))

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)
    
    def vis_train(self, img_train_batch):
        # saving training results
        img_train_batch = torch.cat(img_train_batch, dim=3)
        save_path = os.path.join(self.vis_folder, 'epoch_{:d}_fake.png'.format(self.epoch))
        vis_image = make_grid(self.de_norm(img_train_batch), 1)
        save_image(vis_image, save_path) #, normalize=True)

    def generate(self, image_A, image_B, mask_A=None, mask_B=None, 
                 diff_A=None, diff_B=None, lms_A=None, lms_B=None):
        """image_A is content, image_B is style"""
        with torch.no_grad():
            res = self.G(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B)
        return res

    def test(self, image_A, mask_A, diff_A, lms_A, image_B, mask_B, diff_B, lms_B):        
        with torch.no_grad():
            fake_A = self.generate(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B)
        fake_A = self.de_norm(fake_A)
        fake_A = fake_A.squeeze(0)
        return ToPILImage()(fake_A.cpu())