import torch
import torch.nn as nn
from torch.nn import init, functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
import math


def _pair(inp):
    '''
    Forcefully the input variable if necessary.
    Analog to torch.nn.modules.utils._pair.
    '''
    if isinstance(inp, (list, tuple)):
        return inp
    return (inp, inp)


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
        self.pic = PIC(cfg.PIC_IN_CHANNELS, cfg.PIC_HIDDEN_CHANNELS, cfg.PIC_OUT_CHANNELS, cfg.KERNEL_SIZE)

        self.conv1d = nn.Conv1d(cfg['PIC_OUT_CHANNELS'], 1, 1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.pic(x)
        x = self.conv1d(x)
        return x


class Sender(nn.Module):
    '''
    Sender of address messages.
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
    Recipient of address messages.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super(Receiver, self).__init__()

        kernel_h, kernel_w = _pair(kernel_size)
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


class Modifier(nn.Module):
    '''
    Модификатор адресных сообщений.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 polynomial_degree=3,
                 kernel_size=3,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.polynomial_degree = polynomial_degree
        self.kernel_size = kernel_size

        # self.polynomial_weights = nn.parameter.Parameter(
        #     torch.empty((in_channels, out_channels), **factory_kwargs)
        # )

        if self.polynomial_degree == 1:
            self.activation = nn.ReLU()

        weight_shape = (
            1,
            kernel_size[0],
            kernel_size[1],
            in_channels * polynomial_degree,
            out_channels
        )
        self.weight = Parameter(
            torch.empty(weight_shape, **factory_kwargs)
        )
        if bias:
            bias_shape = (
                1,
                out_channels,
                kernel_size[0],
                kernel_size[1]
            )
            self.bias = Parameter(
                torch.empty(bias_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        tile_dims = [
            x.shape[2] // self.kernel_size[0],
            x.shape[3] // self.kernel_size[1],
        ]
        assert tile_dims[0] * self.kernel_size[0] == x.shape[2]
        assert tile_dims[1] * self.kernel_size[1] == x.shape[3]

        # x = concat([x, x ** 2, ..., x ** polynomial_degree]):
        if self.polynomial_degree > 1:
            exp_x = [x]
            for _ in range(self.polynomial_degree - 1):
                exp_x.append(exp_x[-1] * x)
            x = torch.concat(exp_x, 1)
        elif self.polynomial_degree != 1:
            raise ValueError(
                '`polynomial_degree` must be uint. ' +
                f'Got {self.polynomial_degree}.'
            )

        # x = x @ weight:
        x = x.permute(0, 2, 3, 1)[..., None, :]
        weight = torch.tile(self.weight, [1] + list(tile_dims) + [1, 1])
        x = x @ weight
        assert x.shape[-2] == 1
        x = x[..., 0, :].permute(0, 3, 1, 2)

        # x = x + bias:
        if self.bias is not None:
            bias = torch.tile(self.bias, [1, 1] + list(tile_dims))
            x = x + bias

        if self.polynomial_degree == 1:
            x = self.activation(x)

        return x


class SR_Block(nn.Module):
    def __init__(self, cfg):
        super(SR_Block, self).__init__()

        self.cfg = cfg
        inp_channels = cfg.PIC_IN_CHANNELS
        msg_channels = cfg.PIC_HIDDEN_CHANNELS
        kernel_size = cfg.PIC_KERNEL_SIZE

        # Send
        self.sender = Sender(inp_channels, msg_channels, kernel_size)

        # Conv1x1
        if self.cfg.ADD_CONV1x1:
            self.conv1x1 = nn.Conv2d(msg_channels, msg_channels, 1, 1)

        # Modify
        if self.cfg.MODIFIER:
            self.modifier = Modifier(msg_channels, msg_channels, 1, kernel_size)

        # Global pooling
        global_pools = {
            'avgpool' : nn.AvgPool2d((cfg.IMAGE_SHAPE)),
            'maxpool' : nn.MaxPool2d((cfg.IMAGE_SHAPE)),
        }

        if self.cfg.GLOBAL_POOL != 'none':
            self.global_pool = global_pools[self.cfg.GLOBAL_POOL]

        # Activate
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
        }
        self.activation = activations[cfg.PIC_ACTIVATION]

        # Recieve
        self.receiver = Receiver(msg_channels, inp_channels, kernel_size)

    def forward(self, x):
        if self.cfg.GLOBAL_POOL_PLACE == 'parallel':
            global_features = self.global_pool(x)
        
        x = self.sender(x)
        x = self.activation(x)

        if self.cfg.ADD_CONV1x1:
            x = self.conv1x1(x)
            x = self.activation(x)

        if self.cfg.MODIFIER:
            x = self.modifier(x)

        if self.cfg.GLOBAL_POOL_PLACE == 'serial':
            x = x + self.global_pool(x)
        if self.cfg.GLOBAL_POOL_PLACE == 'parallel':
            x = x + global_features

        x = self.receiver(x)
        
        return x


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.conv_inp = nn.Conv2d(cfg.INPUT_CHANNELS, cfg.PIC_IN_CHANNELS, 3, 1, 1)
        if self.cfg.PIC_NUMBER > 0:
            self.sr_block = SR_Block(cfg)
        self.conv_out = nn.Conv2d(cfg.PIC_IN_CHANNELS, cfg.INPUT_CHANNELS, 3, 1, 1)

    def forward(self, batch_dict):
        outputs = []
        x = batch_dict['im_noisy']
        x = self.conv_inp(x)
        for _ in range(self.cfg.PIC_NUMBER):
            x = self.sr_block(x)
            output = self.conv_out(x)
            outputs.append(output)

        if self.cfg.PIC_NUMBER == 0:
            output = self.conv_out(x)
            outputs.append(output)

        result = batch_dict.copy()
        result.update({
            'outputs': outputs,
        })

        return result

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())