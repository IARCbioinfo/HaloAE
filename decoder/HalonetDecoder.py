
# Inspired by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import torch
from torch import Tensor
import torch.nn as nn
#from .._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import torch
from halonet_pytorch import HaloAttention

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Transposed VGG for images reconstruction
class VGG19Decoder(nn.Module):
    """
    the decoder network
    """
    def __init__(self,  n_layer=3, n_tail_layer = 1, n_first_out_channel = 704):
        super(VGG19Decoder, self).__init__()
        # n_tail_layer == 1 or 2
        # VGG like architecture to reproduce the image. Only the layers used
        # to build the multi-scale feature map are reused.
        # "Pre - VGG" to reduce the number of channels
        self.n_tail_layer = n_tail_layer
        self.n_layer = n_layer
        # Very small VGG
#       self.head_layer = self._make_layer(320, n_first_out_channel) 
        # Small VGG 
        self.head_layer = self._make_layer(704, n_first_out_channel) 
#       self.head_layer = self._make_layer(1408, n_first_out_channel) 
        self.last_out_channel = n_first_out_channel 
        

        self.middel_layer = []
        for i in range(self.n_layer):
            self.middel_layer.append(self._make_layer(self.last_out_channel, int(self.last_out_channel/2)))
            self.last_out_channel = int(self.last_out_channel/2)

        self.middel_layer = nn.Sequential(*self.middel_layer)

        self.tail_layer = []
        if n_tail_layer == 1:
            self.tail_layer1 = self._make_layer(self.last_out_channel, 3, True, scale_factor=4) 
            self.tail_layer.append(self.tail_layer1)
        elif n_tail_layer == 2:
            self.tail_layer1 = self._make_layer(self.last_out_channel, 3, True)
            self.tail_layer2 = self._make_layer(3, 3, True) 
            self.tail_layer.append(self.tail_layer1)
            self.tail_layer.append(self.tail_layer2)
        else:
            assert self.n_tail_layer in [1,2], 'number of tails layers should be equal to 1 or 2'
        self.tail_layer = nn.Sequential(*self.tail_layer)

    def _make_layer(self, inchannel, outchannel, unpool=False, scale_factor=2):

        self.reflecPad_1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_2 = nn.Conv2d(inchannel, outchannel, 3, 1, 0)
        self.relu_1_2 = nn.ReLU(inplace=True)
        if unpool:
            self.unpool_2 = nn.UpsamplingNearest2d(scale_factor=scale_factor)
            return nn.Sequential(self.reflecPad_1_2 ,  self.conv_1_2 , self.relu_1_2, self.unpool_2)
        else:
            return nn.Sequential(self.reflecPad_1_2 ,  self.conv_1_2 , self.relu_1_2)

    def forward(self, input):
        # first block
        out = self.head_layer(input)
        out = self.middel_layer(out)
        out = self.tail_layer(out)
        return out
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def conv5x5T(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv3x3T(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,  padding: int =0, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1T(in_planes: int, out_planes: int, stride: int = 1, padding=0) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding,bias=False)


class Bottleneck(nn.Module):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # HALONET DECODER BLOCK
    # BLOCK ~ equivalent to Halonet encoder block. The convolution layer have been sbustitute by 
    # transposed convolutional layer.

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
        rv: int = 1,
        rb: int =1,
        conv5x5Opt: str='3x3',
        verbose:  bool=False
        ) -> None:
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1T(inplanes, width)
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
            self.conv3 = conv3x3T( round(width), round(planes * self.expansion * rb))
        elif self.conv5x5Opt == '5x5' :
            self.conv3 = conv5x5T( round(width), round(planes * self.expansion * rb))  
        else:
            self.conv3 = conv1x1T( round(width), round(planes * self.expansion * rb))
        self.bn3 = norm_layer( round(planes * self.expansion * rb))
        self.relu = nn.ReLU(inplace=True)
        # 3rd conv transposed layer only if diminution of width and height
        self.conv4 = conv3x3T(round(planes * self.expansion * rb), round(planes * self.expansion * rb), padding = 0, stride=2 )
        self.downsample = downsample
        self.stride = stride
        self.verbose = verbose
         # Padding adjustement 
        self.p1HWd = (0, 1, 0, 1) # Pad the two last dim on one side by one
        self.inplanes =inplanes

    def forward(self, x: Tensor) -> Tensor:
        if self.verbose:
            print(' self.inplanes  ',  self.inplanes)
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
        if self.verbose:
            print('out Conv3 3x3 shape ', out.shape)
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
        if list(out.size())[-1] % 2 != 0 and list(out.size())[-2] % 2 != 0: # If wout and Hout are odd
            out = nn.functional.pad(out, self.p1HWd , mode='reflect')
            identity = nn.functional.pad(identity, self.p1HWd , mode='reflect')
            if self.verbose:
                    print('out  shape after padding ', out.shape)
            if self.verbose:
                print('After padding identity shape ', identity.shape)
                print('After padding out shape ', out.shape)
        # RESIDUAL CONNECTION
        out += identity
        out = self.relu(out)
        if self.verbose:
            print('out  shape ', out.shape)
            print('End of layer ! \n\n')

        return out


class HaloNetT(nn.Module):

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
        ) -> None:
        super(HaloNetT, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.block_size = layers['block_size']
        self.halo_size = layers['halo_size']
        self.rv = layers['rv']
        self.rb = layers['rb']
        self.inplanes = layers['inplanes']
        self.depthL = layers['depthL']
        self.conv5x5Opt = layers['conv5x5']
        self.reduction = layers['reduction'] 
        self.n_vgg_layer = layers['n_vgg_layer']
        self.n_tail_layer_vgg = layers['n_tail_layer_vgg']
        self.n_first_out_channel = layers['n_first_out_channel']
        
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
        # Stage 1
        self.layer1 = self._make_layer(block, self.inplanes*self.depthL[0] ,  layers['stage0'][0], block_size= self.block_size,
                                     halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage0'][1],
                                      rv=self.rv, rb=self.rb , conv5x5Opt= self.conv5x5Opt[0],verbose= False )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stage 2
        self.layer2 = self._make_layer(block, self.inplanes*self.depthL[1],  layers['stage1'][0], block_size= self.block_size,
                                     halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage1'][1], stride=1,
                                       dilate=replace_stride_with_dilation[0],
                                        rv=self.rv, rb=self.rb, conv5x5Opt=self.conv5x5Opt[1], verbose= False )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stage 3
        if self.reduction in ['s4']: # If reduction of the feature map has been set to 4.
            self.layer3 = self._make_layer(block, self.inplanes*self.depthL[2], layers['stage2'][0], block_size= self.block_size,
                             halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage2'][1], stride=2,
                               dilate=replace_stride_with_dilation[0],
                                rv=self.rv, rb=self.rb, conv5x5Opt=self.conv5x5Opt[2], verbose= True )
        else: # No reduction or if the reduction has been set to 2.
            self.layer3 = self._make_layer(block, self.inplanes*self.depthL[2], layers['stage2'][0], block_size= self.block_size,
                             halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage2'][1], stride=1,
                               dilate=replace_stride_with_dilation[0],
                                rv=self.rv, rb=self.rb, conv5x5Opt=self.conv5x5Opt[2], verbose= False )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stage 4
        if self.reduction in ['s2', 's4']:# If reduction of the feature map has been set to 4 or 2.
            self.layer4 = self._make_layer(block, self.inplanes*self.depthL[3], layers['stage3'][0], block_size= self.block_size,
                                         halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage3'][1], stride=2,
                                         dilate=replace_stride_with_dilation[0],
                                          rv=self.rv, rb=self.rb, conv5x5Opt=self.conv5x5Opt[3], verbose= False )
            self.upsampling3 = torch.nn.Upsample(size=(60,60))
        else: # No reduction
            self.layer4 = self._make_layer(block, self.inplanes*self.depthL[3], layers['stage3'][0], block_size= self.block_size,
                                         halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage3'][1], stride=1,
                                         dilate=replace_stride_with_dilation[0],
                                          rv=self.rv, rb=self.rb, conv5x5Opt=self.conv5x5Opt[3], verbose= False )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stage 5
        self.layer5 = self._make_layer(block, self.inplanes*self.depthL[4], layers['stage4'][0], block_size= self.block_size,
                                     halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage4'][1], stride=1,
                                     dilate=replace_stride_with_dilation[0],
                                      rv=self.rv, rb=self.rb, conv5x5Opt=self.conv5x5Opt[4], verbose= False )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Stages summary
        self.L1 = nn.Sequential(self.layer1)
        self.L2 = nn.Sequential(self.layer2)
        self.L3 = nn.Sequential(self.layer3)
        if self.reduction in ['s2', 's4']:
            self.L4 = nn.Sequential(self.layer4, self.upsampling3 )
        else:
            self.L4 = nn.Sequential(self.layer4)

        self.L5 = nn.Sequential(self.layer5)
        self.us_feature_map = torch.nn.Upsample(size=(64,64))
       
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Image reconstruction via a transposed VGG
        self.vgg_decoder = VGG19Decoder(n_layer = self.n_vgg_layer, n_tail_layer = self.n_tail_layer_vgg , n_first_out_channel = self.n_first_out_channel)

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
                     heads: int = 8, rv:float = 1, rb:float = 1 , conv5x5Opt = False, verbose: bool=False) -> nn.Sequential:
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
                conv1x1T(self.inplanes, round(planes * block.expansion * self.rb), stride, padding=0),
                norm_layer(round(planes * block.expansion * self.rb)),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, block_size, halo_size,
                            dim_head, heads, rv, rb,  conv5x5Opt,verbose))
        self.inplanes = round(planes * block.expansion * self.rb)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, block_size = block_size, halo_size =halo_size,
                                 dim_head= dim_head, heads =heads, rv=rv, rb=rb, conv5x5Opt= conv5x5Opt, verbose=verbose))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.L1(x)
        y = self.L2(y)
        y = self.L3(y)
        y = self.L4(y)
        y = self.L5(y)
        feature_map = self.us_feature_map(y) 
        y = self.vgg_decoder(feature_map)
        return y



def _halonet(
    arch: str,
    block: Type[Union[ Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> HaloNetT:
    model = HaloNetT(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model




def halonetT(n_vgg_layer: int = 3, n_tail_layer_vgg: int= 2, n_first_out_channel = 100,
              pretrained: bool = False, progress: bool = True,  reduction: str = 's1', **kwargs: Any) -> HaloNetT:
    r"""Halonet transposed -  Halonet decoder definition
    """
    return _halonet('halonetT', 
        Bottleneck, {
        # Block D : Transposed VGG
        'n_vgg_layer': n_vgg_layer, 
        'n_tail_layer_vgg': n_tail_layer_vgg , 
        'n_first_out_channel': n_first_out_channel,
        # Halonet Decoder
        'reduction':reduction,                     
        'inplanes':29, # Number of channels of the encoded feature map
        'block_size':12, # Size of the blocks
        'halo_size':2, # size of the Halo
        # Definition of the 5 stages: The fisrt element of each tuple is associated with the number of block by stage, the second element with the number of halonet attention heads
        'stage0':(1,4), 'stage1':(1,8), 'stage2':(1,8), 'stage3':(1,8), 'stage4':(1,8),
        # Definition of the filter size of the second convolution layer of each block equaling '3x3' or '5x5' 
        'conv5x5':['3x3','3x3','3x3','5x5','1x1'],   
        # Coefficient of the depth of the encoded feature map at each stage
        'depthL':[1,2.025,2,2.006,2.97],                           
        'rv':1, #  attention output width multiplier
        'rb':1, # bottleneck output width multiplier
         }, 
         pretrained, progress,**kwargs)


if __name__ == '__main__':
    from torchsummary import summary
    model = halonetT(reduction='s1').cuda()
    summary(model, (29, 60, 60))
    x = torch.randn(1,29,60,60).cuda()
    preds = model(x)
    print(preds.size())