import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG as TVGG
from torchvision.models.vgg import load_state_dict_from_url, model_urls, cfgs

from .modules.spectral_norm import spectral_norm as SpectralNorm
from .elegant import Generator


def get_generator(config):
    kwargs = {
        'conv_dim':config.MODEL.G_CONV_DIM,
        'image_size':config.DATA.IMG_SIZE,
        'num_head':config.MODEL.NUM_HEAD,
        'double_encoder':config.MODEL.DOUBLE_E,
        'use_ff':config.MODEL.USE_FF,
        'num_layer_e':config.MODEL.NUM_LAYER_E,
        'num_layer_d':config.MODEL.NUM_LAYER_D,
        'window_size':config.MODEL.WINDOW_SIZE,
        'merge_mode':config.MODEL.MERGE_MODE
    }
    G = Generator(**kwargs)
    return G


def get_discriminator(config):
    kwargs = {
        'input_channel': 3,
        'conv_dim':config.MODEL.D_CONV_DIM,
        'num_layers':config.MODEL.D_REPEAT_NUM,
        'norm':config.MODEL.D_TYPE
    }
    D = Discriminator(**kwargs)
    return D


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, input_channel=3, conv_dim=64, num_layers=3, norm='SN', **unused):
        super(Discriminator, self).__init__()

        layers = []
        if norm=='SN':
            layers.append(SpectralNorm(nn.Conv2d(input_channel, conv_dim, kernel_size=4, stride=2, padding=1)))
        else:
            layers.append(nn.Conv2d(input_channel, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, num_layers):
            if norm=='SN':
                layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        #k_size = int(image_size / np.power(2, repeat_num))
        if norm=='SN':
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, padding=1)))
        else:
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        if norm=='SN':
            self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_makeup = self.conv1(h)
        return out_makeup


class VGG(TVGG):
    def forward(self, x):
        x = self.features(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)