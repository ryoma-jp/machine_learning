import torch
import torch.nn as nn

class BaseConv(nn.Module):
    """
    Conv layer with batch normalization and activation.
    """
    def __init__(self, in_channels, out_channels, ksize, stride, dilation=1, groups=1, use_bn=True, act=None):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            ksize (int): kernel size
            stride (int): stride
            dilation (int): dilation
            groups (int): number of groups
            use_bn (bool): whether to use batch normalization
            act (str): activation function
        """
        super().__init__()
        self.use_bn = use_bn
        self.use_act = act is not None
        pad = (ksize - 1) // 2 if dilation == 1 else dilation * (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation=dilation, groups=groups, bias=not use_bn)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            if act == 'relu':
                self.act = nn.ReLU()
            elif act == 'leaky':
                self.act = nn.LeakyReLU(0.1)
            elif act == 'silu':
                self.act = nn.SiLU()
            else:
                raise ValueError('Unsupported activation function: {}'.format(act))

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x
    
    def fuseforward(self, x):
        """
        Forward method for fused convolution and batch normalization.
        """
        if self.use_bn:
            # Fuse conv and bn
            weight = self.conv.weight * self.bn.weight[:, None, None, None] / torch.sqrt(self.bn.running_var[:, None, None, None] + self.bn.eps)
            bias = self.bn.bias - self.bn.weight * self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            x = nn.functional.conv2d(x, weight, bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        else:
            x = self.conv(x)
        
        if self.use_act:
            x = self.act(x)
        return x

class Bottleneck(nn.Module):
    """
    Bottleneck block for CSPNet.
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=0.5, shortcut=True, depthwise=False, act='silu'):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            stride (int): stride
            expansion (float): expansion ratio
            depthwise (bool): whether to use depthwise separable convolution
            act (str): activation function
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1, act=act)
        if depthwise:
            self.conv2 = nn.Sequential(
                BaseConv(hidden_channels, hidden_channels, 3, stride, groups=hidden_channels, act=act),
                BaseConv(hidden_channels, hidden_channels, 1, 1, act=act)
            )
        else:
            self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride, act=act)
        self.use_res_connect = shortcut and in_channels == out_channels and stride == 1

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.use_res_connect:
            y = x + y
        return y

class SPPBottleneck(nn.Module):
    """
    SPP bottleneck block for CSPNet.
    """
    def __init__(self, in_channels, out_channels, ksizes=(5, 9, 13), act='silu'):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            ksizes (tuple): kernel sizes for pooling
            activation (str): activation function
        """
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in ksizes])
        conv2_channels = hidden_channels * (len(ksizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class CSPLayer(nn.Module):
    """
    CSP layer in YOLOv5.
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act='silu'):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            n (int): number of repeated blocks
            shortcut (bool): whether to use shortcut
            expansion (float): expansion ratio
        """
        super().__init__()
        hidden_channels = int(in_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, 1, act=act)
        self.conv3 = BaseConv(hidden_channels * 2, out_channels, 1, 1, act=act)
        self.convs = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, 1, expansion=1.0, shortcut=shortcut, act=act) for _ in range(n)])
        self.shortcut = shortcut

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y1 = self.convs(y1)
        y = torch.cat((y1, y2), dim=1)
        return self.conv3(y)
