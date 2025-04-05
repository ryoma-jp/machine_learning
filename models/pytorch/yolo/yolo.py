
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
        x = torch.cat([patch_top_left, patch_bottom_left, patch_top_right, patch_bottom_right], dim=1)

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
        out_layers=['dark3', 'dark4', 'dark5'],
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
        self.output_layers = out_layers
        self.width = width
        self.deep_bias = deep_bias
        self.act = act
        self.onnx_export = onnx_export

        # backbone - stem
        in_channels = 3 * 4
        out_channels = in_channels * 2
        self.stem = Focus(in_channels, out_channels, kernel_size=3, stride=1, act=self.act)

        # backbone - dark2
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

        # backbone - dark3
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

        # backbone - dark4
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

        # backbone - dark5
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

        # neck
        neck_channels = [96, 192, 384]
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.neck_conv0 = BaseConv(neck_channels[2], neck_channels[1], 1, 1, act=self.act)
        self.neck_conv1 = BaseConv(neck_channels[1], neck_channels[0], 1, 1, act=self.act)
        self.neck_conv2 = BaseConv(neck_channels[0], neck_channels[0], 3, 2, act=self.act)
        self.neck_conv3 = BaseConv(neck_channels[1], neck_channels[1], 3, 2, act=self.act)
        self.neck_csp0 = CSPLayer(neck_channels[1] * 2, neck_channels[1], n=1, shortcut=False, expansion=0.25, act=self.act)
        self.neck_csp1 = CSPLayer(neck_channels[0] * 2, neck_channels[0], n=1, shortcut=False, expansion=0.25, act=self.act)
        self.neck_csp2 = CSPLayer(neck_channels[0] * 2, neck_channels[1], n=1, shortcut=False, expansion=0.5, act=self.act)
        self.neck_csp3 = CSPLayer(neck_channels[1] * 2, neck_channels[2], n=1, shortcut=False, expansion=0.5, act=self.act)

        # head
        head_channels = [96, 192, 384]
        self.head_stems = nn.ModuleList()
        self.classifier_convs = nn.ModuleList()
        self.regressor_convs = nn.ModuleList()
        self.classifier_preds = nn.ModuleList()
        self.regressor_preds = nn.ModuleList()
        self.object_preds = nn.ModuleList()
        for head_channel in head_channels:
            self.head_stems.append(BaseConv(head_channel, 96, 1, 1, act=self.act))

            self.classifier_convs.append(
                nn.Sequential(*[BaseConv(96, 96, 3, 1, act=self.act), BaseConv(96, 96, 3, 1, act=self.act)])
            )
            self.regressor_convs.append(
                nn.Sequential(*[BaseConv(96, 96, 3, 1, act=self.act), BaseConv(96, 96, 3, 1, act=self.act)])
            )

            self.classifier_preds.append(
                nn.Conv2d(96, self.num_classes, 1, 1, 0)
            )
            self.regressor_preds.append(
                nn.Conv2d(96, 4, 1, 1, 0)
            )
            self.object_preds.append(
                nn.Conv2d(96, 1, 1, 1, 0)
            )


    def forward(self, x):
        # backbone
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x
        x = self.dark3(x)
        outputs['dark3'] = x
        x = self.dark4(x)
        outputs['dark4'] = x
        x = self.dark5(x)
        outputs['dark5'] = x

        backbone_outputs = [outputs[key] for key in self.output_layers]

        # neck
        fpn_out0 = self.neck_conv0(backbone_outputs[2])
        out0_ = self.upsample(fpn_out0)
        out0_ = torch.cat([out0_, backbone_outputs[1]], dim=1)
        out0_ = self.neck_csp0(out0_)

        fpn_out1 = self.neck_conv1(out0_)
        out1_ = self.upsample(fpn_out1)
        out1_ = torch.cat([out1_, backbone_outputs[0]], dim=1)
        neck_out2 = self.neck_csp1(out1_)

        out2_ = self.neck_conv2(neck_out2)
        out2_ = torch.cat([out2_, fpn_out1], dim=1)
        neck_out1 = self.neck_csp2(out2_)

        out3_ = self.neck_conv3(neck_out1)
        out3_ = torch.cat([out3_, fpn_out0], dim=1)
        neck_out0 = self.neck_csp3(out3_)
        neck_out = (neck_out2, neck_out1, neck_out0)

        # head
        head_out = []
        for i, (x, head_stem) in enumerate(zip(neck_out, self.head_stems)):
            head_stem_out = head_stem(x)
            head_classifier_feature = self.classifier_convs[i](head_stem_out)
            head_classifier_out = self.classifier_preds[i](head_classifier_feature)
            head_regressor_feature = self.regressor_convs[i](head_stem_out)
            head_regressor_out = self.regressor_preds[i](head_regressor_feature)
            head_object_out = self.object_preds[i](head_regressor_feature)

            head_out.append(torch.cat([head_regressor_out, head_object_out.sigmoid(), head_classifier_out.sigmoid()], dim=1))
        
        hw_ = [x_.shape[-2:] for x_ in head_out]
        outputs = torch.cat(
            [x_.flatten(start_dim=2) for x_ in head_out], dim=2
        ).permute(0, 2, 1)

        # decode
        grids = []
        strides = []
        strides_=[8, 16, 32]
        for (hsize, wsize), stride in zip(hw_, strides_):
            yv, xv = torch.meshgrid(*[torch.arange(hsize), torch.arange(wsize)], indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(x[0].type())
        strides = torch.cat(strides, dim=1).type(x[0].type())

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)

        return outputs
