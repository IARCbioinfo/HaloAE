# Inspired by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch
from halonet_pytorch import HaloAttention


def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # HALONET ENCODER BLOCK
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block_size: int =8,
        halo_size: int = 4,
        dim_head: int = 32,
        heads: int =4,
        rv: int = 1, # attention output width multiplier
        rb: int =1, # bottleneck output width multiplier
        conv5x5Opt: str='3x3',
        verbose:  bool=False
    ) -> None:
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # First convolution layer
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # Halonet self-attention layer
        self.attn = HaloAttention(
            dim =  width ,         # dimension of feature map
            block_size = block_size,    # neighborhood block size (feature map must be divisible by this)
            halo_size = halo_size,     # halo size (block receptive field)
            dim_head = round( dim_head * rv),   #* rv   # dimension of each head
            heads = heads          # number of attention heads
        )
        # Second convolution layer
        self.conv5x5Opt = conv5x5Opt 
        if self.conv5x5Opt == '3x3':
            self.conv3 = conv3x3( round(width), round(planes * self.expansion * rb))
        elif self.conv5x5Opt == '5x5' :
            self.conv3 = conv5x5( round(width), round(planes * self.expansion * rb))
        else:
            self.conv3 = conv1x1( round(width), round(planes * self.expansion * rb))

        self.bn3 = norm_layer( round(planes * self.expansion * rb))
        self.relu = nn.ReLU(inplace=True)
        # 3rd conv layer only if diminution of width and height
        self.conv4 = conv3x3(round(planes * self.expansion * rb), round(planes * self.expansion * rb), stride=2   )
        self.downsample = downsample
        self.stride = stride
        self.verbose = verbose
        # Padding adjustement 
        self.p1HWd = (0, 1, 0, 1) # Pad the two last dim on one side by one

    def forward(self, x: Tensor) -> Tensor:
        if self.verbose:
            print('self.stride  ', self.stride)
            print('in x ', x.shape)
        out = self.conv1(x)
        identity = x
        out = self.bn1(out)
        out = self.relu(out)
        if self.verbose:
            print('out conV1 shape ', out.shape)
        out  =  self.attn(out)
        if self.verbose:
            print(
              'out attn shape ', out.shape)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.verbose:
            print('out Conv2 shape ', out.shape)

        if self.downsample is not None:
            if self.verbose:
                print('Before downsampling identity shape ', identity.shape)
                print('Before downsampling out shape ', out.shape)
            identity = self.downsample(x)
        if self.stride == 2:
            out = self.conv4(out)

        if self.verbose:
            print('identity shape ', identity.shape)
            print('out shape ', out.shape)
        # RESIDUAL CONNECTION   
        out += identity
        out = self.relu(out)
        if self.verbose:
            print('out  shape ', out.shape)
            print('End of layer ! \n\n')

        return out


class HaloNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[ Bottleneck]],
        layers: dict,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dim_head: int = 16,
        depthL: list = [1,1,1,1],
        squashing_layer: bool = False
            ) -> None:
        super(HaloNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Config
        self._norm_layer = norm_layer
        self.first_conv_in = layers['first_conv_in']
        self.block_size = layers['block_size']
        self.halo_size = layers['halo_size']
        self.rv = layers['rv']
        self.rb = layers['rb']
        self.reduction = layers['reduction']
        self.depthL = layers['depthL'] 
        self.inplanes = layers['inplanes']
        self.conv5x5Opt = layers['conv5x5']
        self.dilation = 1
        self.dim_head = dim_head

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # HEAD LAYER
        self.conv1 = nn.Conv2d(self.first_conv_in, self.inplanes, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.head_halonet = nn.Sequential(self.conv1,
                                          self.bn1,
                                          self.relu )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # BODY
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stage 1
        self.layer1 = self._make_layer(block, int(self.inplanes*self.depthL[0]), layers['stage0'][0], block_size= self.block_size,
                                     halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage0'][1],
                                      rv=self.rv, rb=self.rb , conv5x5Opt= self.conv5x5Opt[0], verbose= False )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stage 2
        if self.reduction in ['s2', 's4']: # Reduction of fm widht and height by a factor 2 or 4
            self.layer2 = self._make_layer(block, int(self.inplanes*self.depthL[1]), layers['stage1'][0], block_size= self.block_size,
                                         halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage1'][1], stride=2,
                                           dilate=replace_stride_with_dilation[0],
                                            rv=self.rv, rb=self.rb, conv5x5Opt= self.conv5x5Opt[1],verbose= False )
            self.upsampling1 = torch.nn.Upsample(size=(36,36))
        else: # No reduction of feature map width and height
            self.layer2 = self._make_layer(block, int(self.inplanes*self.depthL[1]), layers['stage1'][0], block_size= self.block_size,
                                         halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage1'][1], stride=1,
                                           dilate=replace_stride_with_dilation[0],
                                            rv=self.rv, rb=self.rb, conv5x5Opt= self.conv5x5Opt[1],verbose= False )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stage 3
        if self.reduction in ['s4']: # Reduction of fm widht and height or 4
            self.layer3 = self._make_layer(block, int(self.inplanes*self.depthL[2]), layers['stage2'][0], block_size= self.block_size,
                                         halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage2'][1], stride=2,
                                           dilate=replace_stride_with_dilation[0],
                                            rv=self.rv, rb=self.rb, conv5x5Opt= self.conv5x5Opt[2], verbose= False )
            self.upsampling2 = torch.nn.Upsample(size=(12,12))           

        else: # No reduction of feature map width and height or by a factor 2
            self.layer3 = self._make_layer(block, int(self.inplanes*self.depthL[2]), layers['stage2'][0], block_size= self.block_size,
                                         halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage2'][1], stride=1,
                                           dilate=replace_stride_with_dilation[0],
                                            rv=self.rv, rb=self.rb, conv5x5Opt= self.conv5x5Opt[2], verbose= False)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stage 4
        self.layer4 = self._make_layer(block, int(self.inplanes*self.depthL[3]), layers['stage3'][0], block_size= self.block_size,
                                     halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage3'][1], stride=1,
                                     dilate=replace_stride_with_dilation[0],
                                      rv=self.rv, rb=self.rb,conv5x5Opt= self.conv5x5Opt[3], verbose= False )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stages summary
        self.L1 = nn.Sequential( self.layer1)
        if self.reduction in ['s2', 's4']:
            self.L2 = nn.Sequential(self.layer2,
                                        self.upsampling1)
        else:
            self.L2 = nn.Sequential(self.layer2)

        if self.reduction in ['s4']:
            self.L3 = nn.Sequential(self.layer3,
                                        self.upsampling2)
        else:
            self.L3 = nn.Sequential(self.layer3)
        self.L4 = nn.Sequential(self.layer4)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # INITIALISATION
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
                
    def _make_layer(self, block: Type[Union[ Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, block_size:int =8, halo_size: int = 10, dim_head: int = 16,
                     heads: int = 8, rv:float = 1, rb:float = 1 ,  conv5x5Opt = False, verbose: bool=False) -> nn.Sequential:
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Create blocks
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion or self.rb != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, round(planes * block.expansion * self.rb), stride),
                norm_layer(round(planes * block.expansion * self.rb)),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, block_size, halo_size,
                            dim_head, heads, rv, rb,  conv5x5Opt, verbose))
        self.inplanes = round(planes * block.expansion * self.rb)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, block_size = block_size, halo_size =halo_size,
                                 dim_head= dim_head, heads =heads, rv=rv, rb=rb, conv5x5Opt= conv5x5Opt, verbose=verbose))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor): # -> Tensor:
        # See note [TorchScript super()]
        x = self.head_halonet(x)
        x = self.L1(x)
        self.x2 = self.L2(x)
        self.x3 = self.L3(self.x2)
        self.x4 =  self.L4(self.x3)
        x = self.x4
        return x


def _halonet(
    arch: str,
    block: Type[Union[ Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> HaloNet:
    model = HaloNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

#
def halonetE(pretrained: bool = False, progress: bool = True, reduction: str = 's1', **kwargs: Any) -> HaloNet:
    r"""Definiton of halonet encoder
    """
    return _halonet('halonetE', 
        Bottleneck, 
        {'reduction':reduction, # Can equal s1, s2 or s4 : coefficient of reduction of the feature map
        'first_conv_in':704, # Number of channels of the feature map
        'inplanes':int(704/3), # Number of channel after the head layer
        'block_size':12, # Size of the blocks
        'halo_size':2, # size of the Halo
        # Definition of the 4 stages: The fisrt element of each tuple is associated with the number of block by stage, the second element with the number of halonet attention heads
        'stage0':(1,4), 'stage1':(1,8), 'stage2':(1,8), 'stage3':(1,8),
        # Definition of the filter size of the second convolution layer of each block equaling '3x3' or '5x5' 
        'conv5x5':['3x3','3x3', '5x5','5x5'],
        # Coefficient of the depth of the encoded feature map at each stage
        'depthL':[1,.5,.5,.5],
        'rv':1, #  attention output width multiplier
        'rb':1, # bottleneck output width multiplier
        },  
        pretrained, 
        progress,**kwargs)


if __name__ == '__main__':
    from torchsummary import summary
    model = halonetE(reduction='s1').cuda()
    print(model.head_halonet)
    summary(model, (704, 64, 64))
    img = torch.randn(1,704, 64, 64).cuda()
