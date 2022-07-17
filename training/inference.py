from typing import List
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from training.solver import Solver
from training.preprocess import PreProcess
from models.modules.pseudo_gt import expand_area, mask_blend

class InputSample:
    def __init__(self, inputs, apply_mask=None):
        self.inputs = inputs
        self.transfer_input = None
        self.attn_out_list = None
        self.apply_mask = apply_mask

    def clear(self):
        self.transfer_input = None
        self.attn_out_list = None


class Inference:
    """
    An inference wrapper for makeup transfer.
    It takes two image `source` and `reference` in,
    and transfers the makeup of reference to source.
    """
    def __init__(self, config, args, model_path="G.pth"):

        self.device = args.device
        self.solver = Solver(config, args, inference=model_path)
        self.preprocess = PreProcess(config, args.device)
        self.denoise = config.POSTPROCESS.WILL_DENOISE
        self.img_size = config.DATA.IMG_SIZE
        # TODO: can be a hyper-parameter
        self.eyeblur = {'margin': 12, 'blur_size':7}

    def prepare_input(self, *data_inputs):
        """
        data_inputs: List[image, mask, diff, lms]
        """
        inputs = []
        for i in range(len(data_inputs)):
            inputs.append(data_inputs[i].to(self.device).unsqueeze(0))
        # prepare mask
        inputs[1] = torch.cat((inputs[1][:,0:1], inputs[1][:,1:].sum(dim=1, keepdim=True)), dim=1)
        return inputs

    def postprocess(self, source, crop_face, result):
        if crop_face is not None:
            source = source.crop(
                (crop_face.left(), crop_face.top(), crop_face.right(), crop_face.bottom()))
        source = np.array(source)
        result = np.array(result)

        height, width = source.shape[:2]
        small_source = cv2.resize(source, (self.img_size, self.img_size))
        laplacian_diff = source.astype(
            np.float) - cv2.resize(small_source, (width, height)).astype(np.float)
        result = (cv2.resize(result, (width, height)) +
                  laplacian_diff).round().clip(0, 255)

        result = result.astype(np.uint8)

        if self.denoise:
            result = cv2.fastNlMeansDenoisingColored(result)
        result = Image.fromarray(result).convert('RGB')
        return result

    
    def generate_source_sample(self, source_input):
        """
        source_input: List[image, mask, diff, lms]
        """
        source_input = self.prepare_input(*source_input)
        return InputSample(source_input)

    def generate_reference_sample(self, reference_input, apply_mask=None, 
                                  source_mask=None, mask_area=None, saturation=1.0):
        """
        all the operations on the mask, e.g., partial mask, saturation, 
        should be finally defined in apply_mask
        """
        if source_mask is not None and mask_area is not None:
            apply_mask = self.generate_partial_mask(source_mask, mask_area, saturation)
            apply_mask = apply_mask.unsqueeze(0).to(self.device)
        reference_input = self.prepare_input(*reference_input)
        
        if apply_mask is None:
            apply_mask = torch.ones(1, 1, self.img_size, self.img_size).to(self.device)
        return InputSample(reference_input, apply_mask)


    def generate_partial_mask(self, source_mask, mask_area='full', saturation=1.0):
        """
        source_mask: (C, H, W), lip, face, left eye, right eye
        return: apply_mask: (1, H, W)
        """
        assert mask_area in ['full', 'skin', 'lip', 'eye']
        if mask_area == 'full':
            return torch.sum(source_mask[0:2], dim=0, keepdim=True) * saturation
        elif mask_area == 'lip':
            return source_mask[0:1] * saturation
        elif mask_area == 'skin':
            mask_l_eye = expand_area(source_mask[2:3], self.eyeblur['margin']) #* source_mask[1:2]
            mask_r_eye = expand_area(source_mask[3:4], self.eyeblur['margin']) #* source_mask[1:2]
            mask_eye = mask_l_eye + mask_r_eye
            #mask_eye = mask_blend(mask_eye, 1.0, source_mask[1:2], blur_size=self.eyeblur['blur_size'])
            mask_eye = mask_blend(mask_eye, 1.0, blur_size=self.eyeblur['blur_size'])
            return source_mask[1:2] * (1 - mask_eye) * saturation
        elif mask_area == 'eye':
            mask_l_eye = expand_area(source_mask[2:3], self.eyeblur['margin']) #* source_mask[1:2]
            mask_r_eye = expand_area(source_mask[3:4], self.eyeblur['margin']) #* source_mask[1:2]
            mask_eye = mask_l_eye + mask_r_eye
            #mask_eye = mask_blend(mask_eye, saturation, source_mask[1:2], blur_size=self.eyeblur['blur_size'])
            mask_eye = mask_blend(mask_eye, saturation, blur_size=self.eyeblur['blur_size'])
            return mask_eye
  

    @torch.no_grad()
    def interface_transfer(self, source_sample: InputSample, reference_samples: List[InputSample]):
        """
        Input: a source sample and multiple reference samples
        Return: PIL.Image, the fused result
        """
        # encode source
        if source_sample.transfer_input is None:
            source_sample.transfer_input = self.solver.G.get_transfer_input(*source_sample.inputs)
        
        # encode references
        for r_sample in reference_samples:
            if r_sample.transfer_input is None:
                r_sample.transfer_input = self.solver.G.get_transfer_input(*r_sample.inputs, True)

        # self attention
        if source_sample.attn_out_list is None:
            source_sample.attn_out_list = self.solver.G.get_transfer_output(
                    *source_sample.transfer_input, *source_sample.transfer_input
                )
        
        # full transfer for each reference
        for r_sample in reference_samples:
            if r_sample.attn_out_list is None:
                r_sample.attn_out_list = self.solver.G.get_transfer_output(
                    *source_sample.transfer_input, *r_sample.transfer_input
                )

        # fusion
        # if the apply_mask is changed without changing source and references,
        # only the following steps are required
        fused_attn_out_list = []
        for i in range(len(source_sample.attn_out_list)):
            init_attn_out = torch.zeros_like(source_sample.attn_out_list[i], device=self.device)
            fused_attn_out_list.append(init_attn_out)
        apply_mask_sum = torch.zeros((1, 1, self.img_size, self.img_size), device=self.device)
        
        for r_sample in reference_samples:
            if r_sample.apply_mask is not None:
                apply_mask_sum += r_sample.apply_mask
                for i in range(len(source_sample.attn_out_list)):
                    feature_size = r_sample.attn_out_list[i].shape[2]
                    apply_mask = F.interpolate(r_sample.apply_mask, feature_size, mode='nearest')
                    fused_attn_out_list[i] += apply_mask * r_sample.attn_out_list[i]

        # self as reference
        source_apply_mask = 1 - apply_mask_sum.clamp(0, 1)
        for i in range(len(source_sample.attn_out_list)):
            feature_size = source_sample.attn_out_list[i].shape[2]
            apply_mask = F.interpolate(source_apply_mask, feature_size, mode='nearest')
            fused_attn_out_list[i] += apply_mask * source_sample.attn_out_list[i]

        # decode
        result = self.solver.G.decode(
            source_sample.transfer_input[0], fused_attn_out_list
        )
        result = self.solver.de_norm(result).squeeze(0)
        result = ToPILImage()(result.cpu())
        return result

    
    def transfer(self, source: Image, reference: Image, postprocess=True):
        """
        Args:
            source (Image): The image where makeup will be transfered to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transfered image.
        """
        source_input, face, crop_face = self.preprocess(source)
        reference_input, _, _ = self.preprocess(reference)
        if not (source_input and reference_input):
            return None

        #source_sample = self.generate_source_sample(source_input)
        #reference_samples = [self.generate_reference_sample(reference_input)]
        #result = self.interface_transfer(source_sample, reference_samples)
        source_input = self.prepare_input(*source_input)
        reference_input = self.prepare_input(*reference_input)
        result = self.solver.test(*source_input, *reference_input)
        
        if not postprocess:
            return result
        else:
            return self.postprocess(source, crop_face, result)

    def joint_transfer(self, source: Image, reference_lip: Image, reference_skin: Image,
                       reference_eye: Image, postprocess=True):
        source_input, face, crop_face = self.preprocess(source)
        lip_input, _, _ = self.preprocess(reference_lip)
        skin_input, _, _ = self.preprocess(reference_skin)
        eye_input, _, _ = self.preprocess(reference_eye)
        if not (source_input and lip_input and skin_input and eye_input):
            return None

        source_mask = source_input[1]
        source_sample = self.generate_source_sample(source_input)
        reference_samples = [
            self.generate_reference_sample(lip_input, source_mask=source_mask, mask_area='lip'),
            self.generate_reference_sample(skin_input, source_mask=source_mask, mask_area='skin'),
            self.generate_reference_sample(eye_input, source_mask=source_mask, mask_area='eye')
        ]
        
        result = self.interface_transfer(source_sample, reference_samples)
        
        if not postprocess:
            return result
        else:
            return self.postprocess(source, crop_face, result)