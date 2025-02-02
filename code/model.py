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

class Sender(nn.Module):
    '''
    Отправитель адресных сообщений.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super(Sender, self).__init__()
        kwargs['in_channels'] = in_channels
        kwargs['out_channels'] = out_channels
        kwargs['kernel_size'] = kernel_size
        kwargs['stride'] = kernel_size
        kwargs.pop('padding', None)
        kwargs.pop('output_padding', None)
        kwargs.pop('dilation', None)
        self.conv = nn.ConvTranspose2d(**kwargs)

    def forward(self, *args, **kwargs):
        return self.conv(*args, **kwargs)


class Receiver(nn.Module):
    '''
    Получатель адресных сообщений.
    '''
    @staticmethod
    def pair(inp):
        '''
        Принудительно дублирует входную переменную если надо.
        Аналог troch.nn.modules.utils._pair.
        '''
        if isinstance(inp, (list, tuple)):
            return inp
        return (inp, inp)

    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super(Receiver, self).__init__()

        kernel_h, kernel_w = self.pair(kernel_size)
        padding_h = (kernel_h ** 2 - kernel_h * 3) // 2 + 1
        padding_w = (kernel_w ** 2 - kernel_w * 3) // 2 + 1
        kwargs['in_channels'] = in_channels
        kwargs['out_channels'] = out_channels
        kwargs['kernel_size'] = kernel_size
        kwargs['stride'] = kernel_size
        kwargs['padding'] = (padding_h, padding_w)
        kwargs['dilation'] = (kernel_h - 1, kernel_w - 1)
        kwargs.pop('output_padding', None)
        self.conv = nn.Conv2d(**kwargs)

    def forward(self, *args, **kwargs):
        return self.conv(*args, **kwargs)


class SR_Block(nn.Module):
    def __init__(self, cfg):
        super(SR_Block, self).__init__()

        self.cfg = cfg
        inp_channels = cfg.PIC_IN_CHANNELS
        msg_channels = cfg.PIC_HIDDEN_CHANNELS
        kernel_size = cfg.PIC_KERNEL_SIZE
        self.sender = Sender(inp_channels, msg_channels, kernel_size)
        self.receiver = Receiver(msg_channels, inp_channels, kernel_size)

        activations = {
            'relu' : nn.ReLU(),
        }
        self.activation = activations[cfg.PIC_ACTIVATION]

    def forward(self, x):
        x = self.sender(x)
        x = self.activation(x)
        x = self.receiver(x)
        return x
    
    
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.conv_inp = nn.Conv2d(cfg.INPUT_CHANNELS, cfg.PIC_IN_CHANNELS, 3, 1, 1)
        self.block_list = [ SR_Block(cfg) for _ in range(cfg.PIC_NUMBER) ]
        self.conv_out = nn.Conv2d(cfg.PIC_IN_CHANNELS, cfg.INPUT_CHANNELS, 3, 1, 1)



    def forward(self, batch_dict):
        x_steps = []
        x = batch_dict['noisy_im']
        x = self.conv_inp(x)
        for i in range(self.cfg.PIC_NUMBER):
            x = self.block_list[i](x)
            x_steps.append(x)
        x = self.conv_out(x)

        result = batch_dict.copy()
        result.update({
            'x_steps' : x_steps,
            'im_pred' : x,
        })

        return result



        
    
