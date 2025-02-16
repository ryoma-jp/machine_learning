
import torch
import torch.nn as nn

from .core.base_layers import BaseConv, CSPLayer, SPPBottleneck

SUPPORED_ARCHITECTURES = ['yolox_tiny']

class Focus(nn.Module):
    """
    Focus width and height information into channel space.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act='silu'):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): kernel size
            stride (int): stride
            act (str): activation function
        """

        super(Focus, self).__init__()
        self.conv = BaseConv(in_channels, out_channels, kernel_size, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bottom_left = x[..., 1::2, ::2]
        patch_bottom_right = x[..., 1::2, 1::2]
        x = torch.cat([patch_top_left, patch_top_right, patch_bottom_left, patch_bottom_right], dim=1)

        return self.conv(x)

class YOLOX(nn.Module):
    """
    YOLOX model module. The module consists of a backbone, neck and head.
    """
    def __init__(
        self, 
        name, 
        backbone, 
        head, 
        num_classes, 
        width=1.0, 
        deep_bias=False, 
        act='silu', 
        onnx_export=False): 
        """
        Args:
            name (str): name of the model
            backbone (nn.Module): backbone module
            head (nn.Module): head module
            num_classes (int): number of classes
            width (float): width of the backbone
            deep_bias (bool): whether to use deep bias
            act (str): activation function
            onnx_export (bool): whether to export to onnx
        """

        super().__init__()

        self.name = name
        self.backbone = backbone
        self.head = head
        self.num_classes = num_classes
        self.width = width
        self.deep_bias = deep_bias
        self.act = act
        self.onnx_export = onnx_export

        # stem
        in_channels = 3 * 4
        out_channels = in_channels * 2
        self.stem = Focus(in_channels, out_channels, kernel_size=3, stride=1, act=self.act)

        # dark2
        dark2_in_channels = out_channels
        dark2_out_channels = out_channels * 2
        kernel_size = 3
        stride = 2
        print(f'dark2_in_channels: {dark2_in_channels}')
        print(f'dark2_out_channels: {dark2_out_channels}')
        self.dark2 = nn.Sequential(
            BaseConv(dark2_in_channels, dark2_out_channels, kernel_size, stride, act=self.act),
            CSPLayer(dark2_out_channels, dark2_out_channels, n=1, shortcut=True, act=self.act)
        )

        # dark3
        dark3_in_channels = dark2_out_channels
        dark3_out_channels = dark2_out_channels * 2
        kernel_size = 3
        stride = 2
        print(f'dark3_in_channels: {dark3_in_channels}')
        print(f'dark3_out_channels: {dark3_out_channels}')
        self.dark3 = nn.Sequential(
            BaseConv(dark3_in_channels, dark3_out_channels, kernel_size, stride, act=self.act),
            CSPLayer(dark3_out_channels, dark3_out_channels, n=3, shortcut=True, act=self.act)
        )

        # dark4
        dark4_in_channels = dark3_out_channels
        dark4_out_channels = dark3_out_channels * 2
        kernel_size = 3
        stride = 2
        print(f'dark4_in_channels: {dark4_in_channels}')
        print(f'dark4_out_channels: {dark4_out_channels}')
        self.dark4 = nn.Sequential(
            BaseConv(dark4_in_channels, dark4_out_channels, kernel_size, stride, act=self.act),
            CSPLayer(dark4_out_channels, dark4_out_channels, n=3, shortcut=True, act=self.act)
        )

        # dark5
        dark5_in_channels = dark4_out_channels
        dark5_out_channels = dark4_out_channels * 2
        kernel_size = 3
        stride = 2
        print(f'dark5_in_channels: {dark5_in_channels}')
        print(f'dark5_out_channels: {dark5_out_channels}')
        self.dark5 = nn.Sequential(
            BaseConv(dark5_in_channels, dark5_out_channels, kernel_size, stride, act=self.act),
            SPPBottleneck(dark5_out_channels, dark5_out_channels, act=self.act),
            CSPLayer(dark5_out_channels, dark5_out_channels, n=1, shortcut=False, act=self.act)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        x = self.dark4(x)
        x = self.dark5(x)
        return x