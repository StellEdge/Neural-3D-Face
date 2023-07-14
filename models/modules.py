import torch.nn as nn
import torch
import torch.nn.functional as F

#pointnet input dims: Batchsize, Number of points, Dim of features(first 3 dim must be coordinates)
#pointnet output dims: Batchsize, Number of points, Dim of out features.


#ref image:B H W C
#appearence code: Batchsize, Vector length.
#classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()


#Early fusion MLP:
#Input: PointNet output(B,N,C), Appearence codes(B,W=512)

# class EarlyCodeFusionMLP(nn.Module):
#     #module for early fusion
#     def __init__(self,geo_in_channel,appear_in_channel,out_feature_channel):
#         super(EarlyCodeFusionMLP,self).__init__()
#         self.out_feature_channel = out_feature_channel
#         self.f1 = nn.Linear(geo_in_channel+appear_in_channel,512)
#         self.f2 = nn.Linear(512,256)
#         self.f3 = nn.Linear(256,out_feature_channel)
#     def forward(self,geo_code,app_code):
#
#
#         x = torch.cat((geo_code,app_code),2)
#         x = F.relu(self.f1(x))
#         x = F.relu(self.f2(x))
#         x = F.relu(self.f3(x))
#         return x


# class EarlyCodeFusionMLP(nn.Module):
#     # module for early fusion
#     def __init__(self, geo_in_channel, appear_in_channel, out_feature_channel):
#         super(EarlyCodeFusionMLP, self).__init__()
#         self.out_feature_channel = out_feature_channel
#
#         self.app_bottleneck = nn.Linear(appear_in_channel, 128)
#         self.f1 = nn.Linear(geo_in_channel+128, 256)
#         self.f2 = nn.Linear(256, 128)
#         self.f3 = nn.Linear(128, out_feature_channel)
#         # self.f4 = nn.Linear(64,out_feature_channel)
#
#         self.dropout = nn.Dropout(0.25)
#     def forward(self, geo_code, app_code):
#         app_code = self.app_bottleneck(app_code)
#         app_code = app_code.repeat(1, geo_code.shape[1], 1)
#         x = torch.cat((geo_code, app_code), 2)
#         x = F.relu(self.f1(x))
#         x = self.dropout(x)
#         x = F.relu(self.f2(x))
#         x = self.dropout(x)
#         x = F.sigmoid(self.f3(x))
#         # x = self.dropout(x)
#         # x = F.sigmoid(self.f4(x))
#         #x = F.sigmoid(x)
#         return x
class ANet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ANet, self).__init__()
        channel_counts = [32,64,64,64]
        expansion = 2
        norm_layer = nn.BatchNorm2d
        self.downsample =[nn.Sequential(
                conv1x1(in_channel, channel_counts[0] * expansion, 2),
                norm_layer(channel_counts[0] * expansion),
            ),
            nn.Sequential(
                conv1x1(channel_counts[0] * expansion, channel_counts[1] * expansion, 2),
                norm_layer(channel_counts[1] * expansion),
            )
            ,
            nn.Sequential(
                conv1x1(channel_counts[1] * expansion, channel_counts[2] * expansion, 2),
                norm_layer(channel_counts[2] * expansion),
            ),
            nn.Sequential(
                conv1x1(channel_counts[2] * expansion, channel_counts[2] * expansion, 2),
                norm_layer(channel_counts[3] * expansion),
            ),
            nn.Sequential(
                conv1x1(channel_counts[2] * expansion, out_channel, 2),
                norm_layer(out_channel),
            )]
        layers = []
        layers.append(Bottleneck(inplanes=in_channel, planes=channel_counts[0] , stride=2, downsample=self.downsample[0] ))
        layers.append(Bottleneck(inplanes=channel_counts[0] * expansion, planes=channel_counts[1], stride=2, downsample=self.downsample[1]))
        layers.append(Bottleneck(inplanes=channel_counts[1] * expansion, planes=channel_counts[2], stride=2, downsample=self.downsample[2]))
        layers.append(Bottleneck(inplanes=channel_counts[2] * expansion, planes=channel_counts[3], stride=2, downsample=self.downsample[3]))
        layers.append(Bottleneck(inplanes=channel_counts[3] * expansion, planes=int(out_channel/expansion), stride=2, downsample=self.downsample[4]))
        self.blocks = nn.Sequential(*layers)
        
        # self.f4 = nn.Linear(64,out_feature_channel)

    def forward(self, x , nce_layers=[0,1,2] , output_medium_features = False):
        if output_medium_features:
            feat = x
            feats = []
            for layer_添加企业成员id, layer in enumerate(self.blocks):
                feat = layer(feat)
                if layer_id in nce_layers:
                        feats.append(feat)
                else:
                    pass
            return feats
        else:
            x=self.blocks(x)
            return x

class PNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PNet, self).__init__()
        channel_counts = [32,64,64,64]
        expansion = 4
        norm_layer = nn.BatchNorm2d
        self.downsample =[nn.Sequential(
                conv1x1(in_channel, channel_counts[0] * expansion, 2),
                norm_layer(channel_counts[0] * expansion),
            ),
            nn.Sequential(
                conv1x1(channel_counts[0] * expansion, channel_counts[1] * expansion, 2),
                norm_layer(channel_counts[1] * expansion),
            )
            ,
            nn.Sequential(
                conv1x1(channel_counts[1] * expansion, channel_counts[2] * expansion, 2),
                norm_layer(channel_counts[2] * expansion),
            ),
            nn.Sequential(
                conv1x1(channel_counts[2] * expansion, channel_counts[2] * expansion, 2),
                norm_layer(channel_counts[3] * expansion),
            ),
            nn.Sequential(
                conv1x1(channel_counts[2] * expansion, out_channel, 2),
                norm_layer(out_channel),
            )]
        layers = []
        layers.append(Bottleneck(inplanes=in_channel, planes=channel_counts[0] , stride=2, downsample=self.downsample[0] ))
        layers.append(Bottleneck(inplanes=channel_counts[0] * expansion, planes=channel_counts[1], stride=2, downsample=self.downsample[1]))
        layers.append(Bottleneck(inplanes=channel_counts[1] * expansion, planes=channel_counts[2], stride=2, downsample=self.downsample[2]))
        layers.append(Bottleneck(inplanes=channel_counts[2] * expansion, planes=channel_counts[3], stride=2, downsample=self.downsample[3]))
        layers.append(Bottleneck(inplanes=channel_counts[3] * expansion, planes=int(out_channel/expansion), stride=2, downsample=self.downsample[4]))
        self.blocks = nn.Sequential(*layers)
        # self.f4 = nn.Linear(64,out_feature_channel)

    def forward(self, x , nce_layers=[0,1,2] , output_medium_features = False):
        if output_medium_features:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.blocks):
                feat = layer(feat)
                if layer_id in nce_layers:
                        feats.append(feat)
                else:
                    pass
            return feats
        else:
            x=self.blocks(x)
            return x

class PNet_NoRes(nn.Module):
    def __init__(self, in_channel, out_channel =512):
        super(PNet_NoRes, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.downsample =nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
        # self.f4 = nn.Linear(64,out_feature_channel)

    def forward(self, x ):
        x=self.downsample(x)
        return x

class Dummy_Discriminator(nn.Module):
    def __init__(self):
        super(Dummy_Discriminator, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # 4*4*512
        # self.f4 = nn.Linear(64,out_feature_channel)
        self.linear = nn.Linear(4 * 4 * 512, 1)

    def forward(self, x):
        N = x.shape[0]
        x = self.downsample(x)  # (N,4,4,512)
        x_flat = x.reshape(N, -1)
        # print(x_flat.shape)
        x_flat = self.linear(x_flat)
        return x_flat



class ANet_NoRes(nn.Module):
    def __init__(self, in_channel, style_dim =512):
        super(ANet_NoRes, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.downsample =nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1,bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )#4*4*512
        # self.f4 = nn.Linear(64,out_feature_channel)
        self.style_dim = style_dim
        self.linear = nn.Linear(4*4*512,18*style_dim)
    def forward(self, x ):
        N = x.shape[0]
        x = self.downsample(x) #(N,4,4,512)
        x_flat = x.reshape(N,-1)
        # print(x_flat.shape)
        x_flat = self.linear(x_flat)
        out = torch.reshape(x_flat,(N,18,self.style_dim))   #self.transfer_out_code.unsqueeze(1).repeat(1, 18, 1)
        return out


class EarlyCodeFusionMLP(nn.Module):
    # module for early fusion
    def __init__(self, geo_in_channel, appear_in_channel, out_feature_channel):
        super(EarlyCodeFusionMLP, self).__init__()
        self.out_feature_channel = out_feature_channel

        self.app_bottleneck = nn.Linear(appear_in_channel, 128)
        self.f1 = nn.Linear(geo_in_channel + 128, 256)
        self.f2 = nn.Linear(256, 128)
        self.f3 = nn.Linear(128, out_feature_channel)
        # self.f4 = nn.Linear(64,out_feature_channel)

    def forward(self, geo_code, app_code):
        app_code = self.app_bottleneck(app_code)
        app_code = app_code.repeat(1, geo_code.shape[1], 1)
        x = torch.cat((geo_code, app_code), 2)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return x

class EarlyCodeFusionMLP_V3(nn.Module):
    # module for early fusion
    def __init__(self, geo_in_channel, appear_in_channel, out_feature_channel):
        super(EarlyCodeFusionMLP_V3, self).__init__()
        self.out_feature_channel = out_feature_channel

        # self.app_bottleneck = nn.Linear(appear_in_channel, 128)
        self.f1 = nn.Linear(geo_in_channel + appear_in_channel, 256)
        self.f2 = nn.Linear(256, 128)
        self.f3 = nn.Linear(128, out_feature_channel)
        # self.f4 = nn.Linear(64,out_feature_channel)

    def forward(self, geo_code, app_code):
        # app_code = self.app_bottleneck(app_code)
        app_code = app_code.repeat(1, geo_code.shape[1], 1)
        x = torch.cat((geo_code, app_code), 2)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return x

class EarlyCodeFusionMLP_V4(nn.Module):
    # module for early fusion
    def __init__(self, geo_in_channel, appear_in_channel, out_feature_channel):
        super(EarlyCodeFusionMLP_V4, self).__init__()
        self.out_feature_channel = out_feature_channel

        self.app_bottleneck = nn.Linear(appear_in_channel, 256)
        self.f1 = nn.Linear(geo_in_channel + 256, 512)
        self.f2 = nn.Linear(512, 256)
        self.f3 = nn.Linear(256, out_feature_channel)
        # self.f4 = nn.Linear(64,out_feature_channel)

    def forward(self, geo_code, app_code):
        app_code = self.app_bottleneck(app_code)
        app_code = app_code.repeat(1, geo_code.shape[1], 1)
        x = torch.cat((geo_code, app_code), 2)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return x

class LateCodeFusion(nn.Module):
    # module for early fusion
    def __init__(self, in_channels, out_channels):
        super(LateCodeFusion, self).__init__()

        self.f1 = nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.f2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.f3 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        # self.f4 = nn.Linear(64,out_feature_channel)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return x

def load_networks(self, epoch):
    """Load all the networks from the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    if self.opt.isTrain and self.opt.pretrained_name is not None:
        load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
    else:
        load_dir = self.save_dir
    load_filename = 'epoch_%s.pth' % (epoch)
    load_path = os.path.join(load_dir, load_filename)
    state_dict = torch.load(load_path, map_location=self.device)
    print('loading the model from %s' % load_path)

    for name in self.model_names:
        if isinstance(name, str):
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            net.load_state_dict(state_dict[name])

    if self.opt.phase != 'test':
        if self.opt.continue_train:
            print('loading the optim from %s' % load_path)
            for i, optim in enumerate(self.optimizers):
                optim.load_state_dict(state_dict['opt_%02d' % i])

            try:
                print('loading the sched from %s' % load_path)
                for i, sched in enumerate(self.schedulers):
                    sched.load_state_dict(state_dict['sched_%02d' % i])
            except:
                print('Failed to load schedulers, set schedulers according to epoch count manually')
                for i, sched in enumerate(self.schedulers):
                    sched.last_epoch = self.opt.epoch_count - 1

from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # print(out.shape)
        # print(identity.shape)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            use_last_fc: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.use_last_fc = use_last_fc
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.use_last_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.use_last_fc:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

func_dict = {
    'resnet50': (resnet50, 2048),
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

class ReconNetWrapper(nn.Module):
    fc_dim=257
    def __init__(self, net_recon, use_last_fc=False, init_path=None):
        super(ReconNetWrapper, self).__init__()
        self.use_last_fc = use_last_fc
        func, last_dim = func_dict['resnet50']
        backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)
        if init_path and os.path.isfile(init_path):
            state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
            backbone.load_state_dict(state_dict)
            print("loading init net_recon %s from %s" %(net_recon, init_path))
        self.backbone = backbone
        if not use_last_fc:
            self.final_layers = nn.ModuleList([
                conv1x1(last_dim, 80, bias=True), # id layer
                conv1x1(last_dim, 64, bias=True), # exp layer
                conv1x1(last_dim, 80, bias=True), # tex layer
                conv1x1(last_dim, 3, bias=True),  # angle layer
                conv1x1(last_dim, 27, bias=True), # gamma layer
                conv1x1(last_dim, 2, bias=True),  # tx, ty
                conv1x1(last_dim, 1, bias=True)   # tz
            ])
            for m in self.final_layers:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        return x

class AppearCode3DMM(nn.Module):
    def __init__(self):
        super(AppearCode3DMM, self).__init__()
        self.net_recon = ReconNetWrapper("net_recon")
        load_filename = 'models/epoch_20.pth'
        state_dict = torch.load(load_filename, map_location= torch.device('cpu') )
        # for keys in state_dict['net_recon']:
        #     print(keys, "\t", state_dict['net_recon'][keys].size())
        self.net_recon.load_state_dict(state_dict['net_recon'])

    def forward(self,input_img):
        output_coeff = self.net_recon(input_img)
        tex_coeffs = output_coeff[:, 144: 224]
        return tex_coeffs

    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

# if __name__ == '__main__':
#     pnet = PNet(128,512)
#     x = torch.randn(10,128,256,256)
#     x = pnet(x)
#     print(x.shape)