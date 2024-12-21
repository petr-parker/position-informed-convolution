import torch
import torch.nn as nn
import torch.nn.functional as F

class PIC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size):
        super(PIC, self).__init__()
        assert kernel_size % 2 == 1, 'Kernel Size must be odd!'
        # self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=kernel_size)
        # self.activation = torch.nn.Sigmoid()
        # self.conv = nn.Conv2d(, out_channels, kernel_size, stride=kernel_size, dilation=kernel_size//2, padding=kernel_size//2)

        self.pad00 = torch.nn.ZeroPad2d((0, 2, 0, 2))
        self.conv00 = nn.Conv2d(in_channels, hidden_channels, 1, 1)
        
        self.pad01 = torch.nn.ZeroPad2d((1, 1, 0, 2))
        self.conv01 = nn.Conv2d(in_channels, hidden_channels, 1, 1)

        self.pad02 = torch.nn.ZeroPad2d((2, 0, 0, 2))
        self.conv02 = nn.Conv2d(in_channels, hidden_channels, 1, 1)

        self.pad10 = torch.nn.ZeroPad2d((0, 2, 1, 1))
        self.conv10 = nn.Conv2d(in_channels, hidden_channels, 1, 1)

        self.pad11 = torch.nn.ZeroPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(in_channels, hidden_channels, 1, 1)

        self.pad12 = torch.nn.ZeroPad2d((2, 0, 1, 1))
        self.conv12 = nn.Conv2d(in_channels, hidden_channels, 1, 1)

        self.pad20 = torch.nn.ZeroPad2d((0, 2, 2, 0))
        self.conv20 = nn.Conv2d(in_channels, hidden_channels, 1, 1)

        self.pad21 = torch.nn.ZeroPad2d((1, 1, 2, 0))
        self.conv21 = nn.Conv2d(in_channels, hidden_channels, 1, 1)
        
        self.pad22 = torch.nn.ZeroPad2d((2, 0, 2, 0))
        self.conv22 = nn.Conv2d(in_channels, hidden_channels, 1, 1)
        
        self.activation = nn.Sigmoid()

        self.final = nn.Conv2d(hidden_channels * 9, out_channels, 1, 1)

    def forward(self, x):
        x00 = self.pad00(self.conv00(x))
        x01 = self.pad01(self.conv01(x))
        x02 = self.pad02(self.conv02(x))
        x10 = self.pad10(self.conv10(x))
        x11 = self.pad11(self.conv11(x))
        x12 = self.pad12(self.conv12(x))
        x20 = self.pad20(self.conv20(x))
        x21 = self.pad21(self.conv21(x))
        x22 = self.pad22(self.conv22(x))
        x = torch.cat([x00, x01, x02, x10, x11, x12, x20, x21, x22], dim=1)
        x = x[..., 1:-1, 1:-1]
        x = self.final(x)
        x = self.activation(x)
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
    

