# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from inspect import getmembers
from FeatureExtraction.features_extractor import Extractor


class HaloAE_model(nn.Module):
    def __init__(self,  
                 Init= True, 
                 # For block D
                 n_layer_vgg = 3, 
                 n_tail_layer_vgg=2, 
                 first_out_channel = 100,
                 # For block E
                 num_classes=2,
                 img_size = 256,
                 batch_size = 8,
                 FM=False, # remove block D if True
                 logit = True,# remove block E if True
                ):


        super(HaloAE_model, self).__init__()
        self.Init = Init  
        self.logit = logit
        self.img_size = img_size
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Block A : Multi scale feature map extractor
        vgg19_layers = ('relu1_1', 'relu2_1',  'relu3_1', 'relu3_4')
        
        self.feature_extractor = Extractor(backbone="vgg19",
                              cnn_layers=vgg19_layers,
                              featmap_size=(self.img_size, self.img_size),
                              device='cuda:0',
                            )
        self.feature_extractor.to('cuda:0')

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Block C : Halonet autoencoder for feature map reconstruction
        import encoder.HalonetEncoder as HalonetEnc
        import decoder.HalonetDecoder as HalonetDec
        # Halonet Encoder
        self.halonetE = HalonetEnc.halonetE( )
        self.halonetE_head_halonet = self.halonetE.head_halonet.to('cuda:0')
        self.halonetE_l1 = self.halonetE.L1.to('cuda:0')
        self.halonetE_l2 = self.halonetE.L2.to('cuda:0')
        self.halonetE_l3 = self.halonetE.L3.to('cuda:0')
        self.halonetE_l4 = self.halonetE.L4.to('cuda:0')
        # Halonet Decoder + TVGG ie Block D
        self.HalonetD = HalonetDec.halonetT(n_vgg_layer = n_layer_vgg,
                                             n_tail_layer_vgg = n_tail_layer_vgg, 
                                             n_first_out_channel = first_out_channel)
        self.HalonetD_l1 =  self.HalonetD.L1.to('cuda:0')
        self.HalonetD_l2 =  self.HalonetD.L2.to('cuda:0')
        self.HalonetD_l3 =  self.HalonetD.L3.to('cuda:0')
        self.HalonetD_l4 =  self.HalonetD.L4.to('cuda:0')
        self.HalonetD_l5 =  self.HalonetD.L5.to('cuda:0')
        self.HalonetD_upfm =  self.HalonetD.us_feature_map.to('cuda:0') # Feature map upsampling 60x60x704 => 64x64x704
        # T-VGG ir Block D
        self.FM = FM
        if not self.FM:
            # Network with image reconstruction
            self.HalonetD_TVGG  = self.HalonetD.vgg_decoder.to('cuda:0')
            if self.logit: # If classif. layer ie. Block E
                self.out = nn.Linear(img_size*img_size*3, num_classes).to('cuda:0')
        else:
            self.out = nn.Linear(704*64*64, num_classes).to('cuda:0')
             
        if self.Init:
            print("\nInitializing network weights.........")
            initialize_weights(self.halonetE, self.HalonetD)
        
    def forward(self,x):
        b = x.size(0)
        # Feature map generation
        feature_map = self.feature_extractor(x)
        # Feature map encoding through Halo encoder
        encoded_FMCNN = self.halonetE_head_halonet(feature_map)
        encodedFM_l1 = self.halonetE_l1(encoded_FMCNN)
        encodedFM_l2 = self.halonetE_l2(encodedFM_l1)
        encodedFM_l3 = self.halonetE_l3(encodedFM_l2)
        encodedFM_l4 = self.halonetE_l4(encodedFM_l3)
        # Feature map decoding through Halo decoder
        decodedFM_l1 = self.HalonetD_l1(encodedFM_l4)
        decodedFM_l2 = self.HalonetD_l2(decodedFM_l1)
        decodedFM_l2 = decodedFM_l2.to('cuda:0')
        decodedFM_l3 = self.HalonetD_l3(decodedFM_l2)
        decodedFM_l4 = self.HalonetD_l4(decodedFM_l3)
        decodedFM_l5 = self.HalonetD_l5(decodedFM_l4)
        decodedFM_l5 = decodedFM_l5.to('cuda:0')
        decodedFM = self.HalonetD_upfm(decodedFM_l5) # \hat{fm} => Feature map reconstructed
        # Image reconstruction
        if not self.FM:
            decodedFM = decodedFM.to('cuda:0')
            decodedIm = self.HalonetD_TVGG(decodedFM)
            if  self.logit:
                logit = self.out(torch.flatten(decodedIm,1,-1))
                return logit, feature_map, decodedIm, decodedFM # Full model
            else:
                return  feature_map, decodedIm, decodedFM # No classification
        
        else: # No block D
            logit = self.out(torch.flatten(decodedFM,1,-1))
            if  self.logit:
                return logit, feature_map,  decodedFM # No Image reconstruction
            else:
                return feature_map,  decodedFM # Neither classification nor image reconstruction
# Initialize weight function
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


if __name__ == "__main__":
    from torchsummary import summary
    mod = HaloAE_model()
    print(mod)
    print(summary(mod, (3,256,256)))
    print('\n\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')
    print(mod.out)



