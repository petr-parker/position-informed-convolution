import torch
import torch.nn as nn
import torch.nn.functional as F

class PIC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size):
        super(PIC, self).__init__()
        assert kernel_size % 2 == 1, 'Kernel Size must be odd!'
        self.up = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size, stride=1)
        self.conv = nn.Conv2d(hidden_channels, out_channels, kernel_size, stride=kernel_size, dilation=kernel_size//2, padding=kernel_size//2)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class SimpleModel(nn.Module):
    def __init__(self, cfg):
        self.conv2d = nn.Conv2d(1, cfg.PIC_IN_CHANNELS, 3, 1, 1)
        self.pic = PIC(cfg['PIC_IN_CHANNELS'], cfg['PIC_HIDDEN_CHANNELS'], cfg['PIC_OUT_CHANNELS'], cfg['KERNEL_SIZE'])

        self.conv1d = nn.Conv1d(cfg['PIC_OUT_CHANNELS'], 1, 1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.pic(x)
        x = self.conv1d(x)
        return x
    

