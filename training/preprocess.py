import os
import sys
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional
sys.path.append('.')

import faceutils as futils
from training.config import get_config

class PreProcess:

    def __init__(self, config, need_parser=True, device='cpu'):
        self.img_size = config.DATA.IMG_SIZE   
        self.device = device

        xs, ys = np.meshgrid(
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            ),
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            )
        )
        xs = xs[None].repeat(config.PREPROCESS.LANDMARK_POINTS, axis=0)
        ys = ys[None].repeat(config.PREPROCESS.LANDMARK_POINTS, axis=0)
        fix = np.concatenate([ys, xs], axis=0) 
        self.fix = torch.Tensor(fix) #(136, h, w)
        if need_parser:
            self.face_parse = futils.mask.FaceParser(device=device)

        self.up_ratio    = config.PREPROCESS.UP_RATIO
        self.down_ratio  = config.PREPROCESS.DOWN_RATIO
        self.width_ratio = config.PREPROCESS.WIDTH_RATIO
        self.lip_class   = config.PREPROCESS.LIP_CLASS
        self.face_class  = config.PREPROCESS.FACE_CLASS
        self.eyebrow_class  = config.PREPROCESS.EYEBROW_CLASS
        self.eye_class  = config.PREPROCESS.EYE_CLASS

        self.transform = transforms.Compose([
            transforms.Resize(config.DATA.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    
    ############################## Mask Process ##############################
    # mask attribute: 0:background 1:face 2:left-eyebrow 3:right-eyebrow 4:left-eye 5: right-eye 6: nose
    # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
    def mask_process(self, mask: torch.Tensor):
        '''
        mask: (1, h, w)
        '''        
        mask_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_face = (mask == self.face_class[0]).float() + (mask == self.face_class[1]).float()

        #mask_eyebrow_left = (mask == self.eyebrow_class[0]).float()
        #mask_eyebrow_right = (mask == self.eyebrow_class[1]).float()
        mask_face += (mask == self.eyebrow_class[0]).float()
        mask_face += (mask == self.eyebrow_class[1]).float()

        mask_eye_left = (mask == self.eye_class[0]).float()
        mask_eye_right = (mask == self.eye_class[1]).float()

        #mask_list = [mask_lip, mask_face, mask_eyebrow_left, mask_eyebrow_right, mask_eye_left, mask_eye_right]
        mask_list = [mask_lip, mask_face, mask_eye_left, mask_eye_right]
        mask_aug = torch.cat(mask_list, 0) # (C, H, W)
        return mask_aug      

    def save_mask(self, mask: torch.Tensor, path):
        assert mask.shape[0] == 1
        mask = mask.squeeze(0).numpy().astype(np.uint8)
        mask = Image.fromarray(mask)
        mask.save(path)

    def load_mask(self, path):
        mask = np.array(Image.open(path).convert('L'))
        mask = torch.FloatTensor(mask).unsqueeze(0)
        mask = functional.resize(mask, self.img_size, transforms.InterpolationMode.NEAREST)
        return mask
    
    ############################## Landmarks Process ##############################
    def lms_process(self, image:Image):
        face = futils.dlib.detect(image)
        # face: rectangles, List of rectangles of face region: [(left, top), (right, bottom)]
        if not face:
            return None
        face = face[0]
        lms = futils.dlib.landmarks(image, face) * self.img_size / image.width # scale to fit self.img_size
        # lms: narray, the position of 68 key points, (68 ,2)
        lms = torch.IntTensor(lms.round()).clamp_max_(self.img_size - 1)
        # distinguish upper and lower lips 
        lms[61:64,0] -= 1; lms[65:68,0] += 1
        for i in range(3):
            if torch.sum(torch.abs(lms[61+i] - lms[67-i])) == 0:
                lms[61+i,0] -= 1;  lms[67-i,0] += 1
        # double check
        '''for i in range(48, 67):
            for j in range(i+1, 68):
                if torch.sum(torch.abs(lms[i] - lms[j])) == 0:
                    lms[i,0] -= 1; lms[j,0] += 1'''
        return lms       
    
    def diff_process(self, lms: torch.Tensor, normalize=False):
        '''
        lms:(68, 2)
        '''
        lms = lms.transpose(1, 0).reshape(-1, 1, 1) # (136, 1, 1)
        diff = self.fix - lms # (136, h, w)

        if normalize:
            norm = torch.norm(diff, dim=0, keepdim=True).repeat(diff.shape[0], 1, 1)
            norm = torch.where(norm == 0, torch.tensor(1e10), norm)
            diff /= norm
        return diff

    def save_lms(self, lms: torch.Tensor, path):
        lms = lms.numpy()
        np.save(path, lms)
    
    def load_lms(self, path):
        lms = np.load(path)
        return torch.IntTensor(lms)

    ############################## Compose Process ##############################
    def preprocess(self, image: Image, is_crop=True):
        '''
        return: image: Image, (H, W), mask: tensor, (1, H, W)
        '''
        face = futils.dlib.detect(image)
        # face: rectangles, List of rectangles of face region: [(left, top), (right, bottom)]
        if not face:
            return None, None, None

        face_on_image = face[0]
        if is_crop:
            image, face, crop_face = futils.dlib.crop(
                image, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        else:
            face = face[0]; crop_face = None
        # image: Image, cropped face
        # face: the same as above
        # crop face: rectangle, face region in cropped face
        np_image = np.array(image) # (h', w', 3)

        mask = self.face_parse.parse(cv2.resize(np_image, (512, 512))).cpu()
        # obtain face parsing result
        # mask: Tensor, (512, 512)
        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (self.img_size, self.img_size),
            mode="nearest").squeeze(0).long() #(1, H, W)

        lms = futils.dlib.landmarks(image, face) * self.img_size / image.width # scale to fit self.img_size
        # lms: narray, the position of 68 key points, (68 ,2)
        lms = torch.IntTensor(lms.round()).clamp_max_(self.img_size - 1)
        # distinguish upper and lower lips 
        lms[61:64,0] -= 1; lms[65:68,0] += 1
        for i in range(3):
            if torch.sum(torch.abs(lms[61+i] - lms[67-i])) == 0:
                lms[61+i,0] -= 1;  lms[67-i,0] += 1

        image = image.resize((self.img_size, self.img_size), Image.ANTIALIAS)
        return [image, mask, lms], face_on_image, crop_face
    
    def process(self, image: Image, mask: torch.Tensor, lms: torch.Tensor):
        image = self.transform(image)
        mask = self.mask_process(mask)
        diff = self.diff_process(lms)
        return [image, mask, diff, lms]
    
    def __call__(self, image:Image, is_crop=True):
        source, face_on_image, crop_face = self.preprocess(image, is_crop)
        if source is None:
            return None, None, None
        return self.process(*source), face_on_image, crop_face


if __name__ == "__main__":
    config = get_config()
    preprocessor = PreProcess(config, device='cuda:0')
    if not os.path.exists(os.path.join(config.DATA.PATH, 'lms')):
        os.makedirs(os.path.join(config.DATA.PATH, 'lms', 'makeup'))
        os.makedirs(os.path.join(config.DATA.PATH, 'lms', 'non-makeup'))
    
    # process makeup images
    print("Processing makeup images...")
    with open(os.path.join(config.DATA.PATH, 'makeup.txt'), 'r') as f:
        for line in f.readlines():
            img_name = line.strip()
            raw_image = Image.open(os.path.join(config.DATA.PATH, 'images', img_name)).convert('RGB')
            lms = preprocessor.lms_process(raw_image)
            if lms is not None:
                base_name = os.path.splitext(img_name)[0]
                preprocessor.save_lms(lms, os.path.join(config.DATA.PATH, 'lms', f'{base_name}.npy'))
    print("Done.")

    # process non-makeup images
    print("Processing non-makeup images...")
    with open(os.path.join(config.DATA.PATH, 'non-makeup.txt'), 'r') as f:
        for line in f.readlines():
            img_name = line.strip()
            raw_image = Image.open(os.path.join(config.DATA.PATH, 'images', img_name)).convert('RGB')
            lms = preprocessor.lms_process(raw_image)
            if lms is not None:
                base_name = os.path.splitext(img_name)[0]
                preprocessor.save_lms(lms, os.path.join(config.DATA.PATH, 'lms', f'{base_name}.npy'))
    print("Done.")
    