# -*- coding: utf-8 -*-

import torch
import torchvision.utils as utils
from torchvision import transforms
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


import pytorch_ssim
import  dataset.MVTECDataLoader  as MVTECDataLoader
from HaloAE import HaloAE_model as ae
from dataset.cutpaste import CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Argparse declaration 
    
ap = argparse.ArgumentParser()
## Optimisation
ap.add_argument("-e", "--epochs", required=False, default= 251, type = int,help="Number of training epochs.")
ap.add_argument("-lr", "--learning_rate", required=False, default= 0.0001, help="learning rate")
ap.add_argument("-b", "--batch_size", required=False, default= 4, help= "batch size")

## Resources
ap.add_argument("-w", "--workers", required=False, default= 4, help= "Nb process")

## Model
ap.add_argument("-FM", "--FMOnly", required=False, default= False,  type=bool, help=" if FMOnly == True no reconstruction of images (ie no Block D).")
ap.add_argument("-nl_vgg", "--n_layer_vgg", required=False, default= 4, type=int, help="Number of layers in the transposed VGG network (Block D).")
ap.add_argument("-nfoc_vgg", "--n_first_out_channels_vgg", required=False, default= 380, type=int, help="Number of out channel first VGG layer (Block D)")

## Checkpoint
ap.add_argument("-mnc", "--model_name_checkpoint", required=True, help="Checkpoint filename")
ap.add_argument("-check_path", "--checkpoint_path", required=True,  help="os.path.join(checkpoint_path, model_name_checkpoint => Where the model will be saved" )
ap.add_argument("-wi", "--weight_initialisation", required=False, default= 'None',  help="Path to a model to initialize the weights.") # None == random initialization
ap.add_argument("-rd", "--run_directory", required=True, help="Run directory name")
ap.add_argument("-s", "--summury_path", required=True, help="os.path.join(summury_path, run_directory) => Tensorboard logdir.")

## Data
ap.add_argument("-obj", "--MVTEC_object", required=True, help="MVTEC object name ")
ap.add_argument("-datadir", "--data_root_directory", required=True, default= 'path/to/MVTEC/object/img/folder',   help="Root directory of the dataset.")

args = vars(ap.parse_args())

# Init. path for tensorboard log, and model checkpoints

full_path_summary = os.path.join(args['summury_path'],args['MVTEC_object'], args['run_directory'])
full_path_chekpoint = os.path.join(args['checkpoint_path'],args['MVTEC_object'])
o_model_name = args['model_name_checkpoint']
os.makedirs(full_path_summary, exist_ok=True)
os.makedirs(full_path_chekpoint, exist_ok=True)
writer = SummaryWriter(log_dir=full_path_summary)

# Init. model hyper-parameters
epoch =int(args["epochs"])
minloss = 1e10
ep =0
ssim_loss = pytorch_ssim.SSIM() # SSIM Loss
loss_cross_ent = torch.nn.CrossEntropyLoss()
size = 256 # Input size
batch_size=int(args["batch_size"])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create a data loader

## Cut & Paste data augmentation
after_cutpaste_transform = transforms.Compose([])
after_cutpaste_transform.transforms.append(transforms.ToTensor())
after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]))
train_transform = transforms.Compose([])#
train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
train_transform.transforms.append(transforms.Resize((size,size)))
# Cut and paste 3 way
train_transform.transforms.append(CutPaste3Way(transform = after_cutpaste_transform))


train_data = MVTECDataLoader.MVTECTrainingDataLoader(root = args['data_root_directory'], 
                                      object_name=args['MVTEC_object'],
                                      transform_process = train_transform
                                        )
train_loader = torch.utils.data.DataLoader(MVTECDataLoader.Repeat(train_data, 3000), batch_size=int(args["batch_size"]), drop_last=True,
                            shuffle=True, num_workers=int(args["workers"]), collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, pin_memory=True, prefetch_factor=5)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create the model

model = ae( Init =  True,
           n_layer_vgg = args['n_layer_vgg'],
           n_tail_layer_vgg = 2,
           first_out_channel = args['n_first_out_channels_vgg'],
           FM = args['FMOnly'])


if args['weight_initialisation'] != 'None':
    model.load_state_dict(torch.load(args['weight_initialisation']), strict=False)
    print('Model loaded ')

model.train()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# loss adaptative weighting

def inv_logistic(x,a,c,d, b=0):
    y= (a-b)/(1+np.exp(x-(c/2))**d) +b
    return y

x = np.arange(0,epoch,1)
a,b,c,d = 1,.4,80,0.07
y1 = inv_logistic(x,a,c,d,b) 
a,b,c,d = -1*10**(-8),1.2,150,0.05
y2 = inv_logistic(x,a,c,d,b) 
a,b,c,d = -1*10**(-8),2,300,0.05
y3 = inv_logistic(x,a,c,d,b) 
y_sum =  y1 + y2 + y3
y1n = y1 / y_sum
y2n = y2 / y_sum
y3n = y3 / y_sum
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Optimizeer
Optimiser = Adam( model.parameters(), lr=float(args["learning_rate"]), weight_decay=0.00001)                        

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training loop
print('The model is training ...')
for i in range(epoch):
    # Follow losses term
    t_loss = []
    loss_classif = []
    loss_IM_MSE = []
    loss_IM_SSIM = []
    loss_FM_MSE = []
    loss_FM_SSIM = []
    # Adaptative weighting
    a1 = y1n[i]
    a2 = y2n[i]
    a3 = y3n[i]
 
    for c, x in enumerate(train_loader):
        model.zero_grad()
        x = torch.cat(x, axis=0)
        if args['FMOnly']:
            logit, feature_map, reconsFM = model(x.to('cuda:0')) # No block D
        else:
            logit, feature_map, reconsImg, reconsFM = model(x.to('cuda:0'))
        #~~~~~~~~~~~~~ Classification LOSS ~~~~~~~~~~~~~~~~~
    
        logit = logit.cuda(0)

        label = torch.zeros(batch_size*3)
        label[batch_size:] = 1 # Virtually damaged pictures
        label  = label.cuda(0)
        label = label.to(torch.long)
        # Loss
        loss_cls = loss_cross_ent(logit, label)
        loss_classif.append(loss_cls.item())

        #~~~~~~~~~~~~~ Reconsrtuction LOSS ~~~~~~~~~~~~~~~~~
        x  = x.cuda(0)
        img_normal = x[:batch_size] # Original img not "damaged" by Cut&Paste
        reconsFM = reconsFM.cuda(0)
        reconsFM_normal = reconsFM[:batch_size]
        feature_map = feature_map.cuda(0)
        feature_map_normal =  feature_map[:batch_size]
        loss1FM = F.mse_loss(reconsFM_normal, feature_map_normal)
        loss2FM = 1 - pytorch_ssim.ssim(feature_map_normal, reconsFM_normal)
        loss_FM_MSE.append(loss1FM.item())
        loss_FM_SSIM.append(loss2FM.item())
        if args['FMOnly']:
            loss =  a1*loss_cls+ a2*(loss1FM + loss2FM) #+ a4*loss1IM + a5*loss2IM 
                
        else:
            reconsImg = reconsImg.cuda(0)
            reconsImg_normal = reconsImg[:batch_size]
            loss1IM = F.mse_loss(reconsImg_normal, img_normal)
            loss2IM = 1 - pytorch_ssim.ssim(reconsImg_normal, img_normal)
            loss_IM_MSE.append(loss1IM.item())
            loss_IM_SSIM.append(loss2IM.item())
            loss =  a1*loss_cls+  a3*(loss1IM + loss2IM) + a2*(loss1FM + loss2FM)
                   
        t_loss.append(loss.item())
        loss.backward()
        
        Optimiser.step()
    #Tensorboard log
    writer.add_scalar('Classification loss', np.mean(loss_classif), i+1)
    writer.add_scalar('a1', a1, i+1)
        
    writer.add_scalar('SSIM FM loss', np.mean(loss_FM_SSIM), i+1)
    writer.add_scalar('L2 FM loss', np.mean(loss_FM_MSE), i+1)
    writer.add_scalar('a2', a2, i+1)
    
    if not args['FMOnly'] :
        writer.add_scalar('SSIM IM loss', np.mean(loss_IM_SSIM), i+1)
        writer.add_scalar('L2 IM loss', np.mean(loss_IM_MSE), i+1)
        if i%20 == 0:
            c = 0 
            writer.add_image('Reconstructed Image',utils.make_grid(reconsImg), i, dataformats = 'CHW')
        writer.add_scalar('a3', a3, i+1)

    writer.add_scalar('Mean Epoch loss', np.mean(t_loss), i+1)
    print(f'Mean Epoch {i} loss: {np.mean(t_loss)}')
    print(f'Min loss epoch: {i} with min loss: {minloss}')
    writer.close()

    # Saving the best model
    if i%10 == 0 :
        model_name = o_model_name.split('.')[0] + '_' +str(i) + '.pt'
        minloss = np.mean(t_loss)
        ep = i
        print(f'Model saved at epoch {ep}')
        torch.save(model.state_dict(), os.path.join(full_path_chekpoint, model_name))