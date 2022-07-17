import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.histogram_matching import histogram_matching
from .modules.pseudo_gt import fine_align, expand_area, mask_blur


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        target_tensor = target_tensor.expand_as(prediction).to(prediction.device)
        
        loss = self.loss(prediction, target_tensor)
        return loss


def norm(x: torch.Tensor):
    return x * 2 - 1

def de_norm(x: torch.Tensor):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def masked_his_match(image_s, image_r, mask_s, mask_r):
    '''
    image: (3, h, w)
    mask: (1, h, w)
    '''
    index_tmp = torch.nonzero(mask_s)
    x_A_index = index_tmp[:, 1]
    y_A_index = index_tmp[:, 2]
    index_tmp = torch.nonzero(mask_r)
    x_B_index = index_tmp[:, 1]
    y_B_index = index_tmp[:, 2]

    image_s = (de_norm(image_s) * 255) #[-1, 1] -> [0, 255]
    image_r = (de_norm(image_r) * 255)
    
    source_masked = image_s * mask_s
    target_masked = image_r * mask_r
    
    source_match = histogram_matching(
                source_masked, target_masked,
                [x_A_index, y_A_index, x_B_index, y_B_index])
    source_match = source_match.to(image_s.device)
    
    return norm(source_match / 255) #[0, 255] -> [-1, 1]


def generate_pgt(image_s, image_r, mask_s, mask_r, lms_s, lms_r, margins, blend_alphas, img_size=None):
        """
        input_data: (3, h, w)
        mask: (c, h, w), lip, skin, left eye, right eye
        """
        if img_size is None:
            img_size = image_s.shape[1]
        pgt = image_s.detach().clone()

        # skin match
        skin_match = masked_his_match(image_s, image_r, mask_s[1:2], mask_r[1:2])
        pgt = (1 - mask_s[1:2]) * pgt + mask_s[1:2] * skin_match

        # lip match
        lip_match = masked_his_match(image_s, image_r, mask_s[0:1], mask_r[0:1])
        pgt = (1 - mask_s[0:1]) * pgt + mask_s[0:1] * lip_match

        # eye match
        mask_s_eye = expand_area(mask_s[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_s[1:2]
        mask_r_eye = expand_area(mask_r[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_r[1:2]
        eye_match = masked_his_match(image_s, image_r, mask_s_eye, mask_r_eye)
        mask_s_eye_blur = mask_blur(mask_s_eye, blur_size=5, mode='valid')
        pgt = (1 - mask_s_eye_blur) * pgt + mask_s_eye_blur * eye_match

        # tps align
        pgt = fine_align(img_size, lms_r, lms_s, image_r, pgt, mask_r, mask_s, margins, blend_alphas)
        return pgt


class LinearAnnealingFn():
    """
    define the linear annealing function with milestones
    """
    def __init__(self, milestones, f_values):
        assert len(milestones) == len(f_values)
        self.milestones = milestones
        self.f_values = f_values
        
    def __call__(self, t:int):
        if t < self.milestones[0]:
            return self.f_values[0]
        elif t >= self.milestones[-1]:
            return self.f_values[-1]
        else:
            for r in range(len(self.milestones) - 1):
                if self.milestones[r] <= t < self.milestones[r+1]:
                    return ((t - self.milestones[r]) * self.f_values[r+1] \
                            + (self.milestones[r+1] - t) * self.f_values[r]) \
                            / (self.milestones[r+1] - self.milestones[r])


class ComposePGT(nn.Module):
    def __init__(self, margins, skin_alpha, eye_alpha, lip_alpha):
        super(ComposePGT, self).__init__()
        self.margins = margins
        self.blend_alphas = {
            'skin':skin_alpha,
            'eye':eye_alpha,
            'lip':lip_alpha
        }

    @torch.no_grad()
    def forward(self, sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
        pgts = []
        for source, target, mask_src, mask_tar, lms_src, lms_tar in\
            zip(sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
            pgt = generate_pgt(source, target, mask_src, mask_tar, lms_src, lms_tar, 
                               self.margins, self.blend_alphas)
            pgts.append(pgt)
        pgts = torch.stack(pgts, dim=0)
        return pgts   

class AnnealingComposePGT(nn.Module):
    def __init__(self, margins,
            skin_alpha_milestones, skin_alpha_values,
            eye_alpha_milestones, eye_alpha_values,
            lip_alpha_milestones, lip_alpha_values
        ):
        super(AnnealingComposePGT, self).__init__()
        self.margins = margins
        self.skin_alpha_fn = LinearAnnealingFn(skin_alpha_milestones, skin_alpha_values)
        self.eye_alpha_fn = LinearAnnealingFn(eye_alpha_milestones, eye_alpha_values)
        self.lip_alpha_fn = LinearAnnealingFn(lip_alpha_milestones, lip_alpha_values)
        
        self.t = 0
        self.blend_alphas = {}
        self.step()

    def step(self):
        self.t += 1
        self.blend_alphas['skin'] = self.skin_alpha_fn(self.t)
        self.blend_alphas['eye'] = self.eye_alpha_fn(self.t)
        self.blend_alphas['lip'] = self.lip_alpha_fn(self.t)

    @torch.no_grad()
    def forward(self, sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
        pgts = []
        for source, target, mask_src, mask_tar, lms_src, lms_tar in\
            zip(sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
            pgt = generate_pgt(source, target, mask_src, mask_tar, lms_src, lms_tar,
                               self.margins, self.blend_alphas)
            pgts.append(pgt)
        pgts = torch.stack(pgts, dim=0)
        return pgts   


class MakeupLoss(nn.Module):
    """
    Define the makeup loss w.r.t pseudo ground truth
    """
    def __init__(self):
        super(MakeupLoss, self).__init__()

    def forward(self, x, target, mask=None):
        if mask is None:
            return F.l1_loss(x, target)
        else:
            return F.l1_loss(x * mask, target * mask)