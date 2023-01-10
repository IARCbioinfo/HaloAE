# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np
import argparse
import pandas as pd

import pytorch_ssim
from  dataset.MVTECDataLoader import MVTECTestDataLoader
from HaloAE import HaloAE_model as ae

ap = argparse.ArgumentParser()
# Model
ap.add_argument("-FM", "--FMOnly", required=False, default= False,  type=bool, help="FMOnly = True not img")
ap.add_argument("-nl_vgg", "--n_layer_vgg", required=False, default= 4, type=int, help="Number of layer in the transposed VGG network")
ap.add_argument("-nfoc_vgg", "--n_first_out_channels_vgg", required=False, default= 380, type=int, help="Number of out channel first VGG layer")

# Resources
ap.add_argument("-w", "--workers", required=False, default=4, help= "Nb process")

# Data
ap.add_argument("-b", "--batch_size", required=False, default=1, help= "batch size")
ap.add_argument("-obj", "--MVTEC_object", required=False, default= 'carpet' , help="MVTEC object name ")
ap.add_argument("-datadir", "--data_root_directory", required=False, default= '/path/to/MVTEC', help="Root directory of the dataset")

# Loading and saving
ap.add_argument("-check_path", "--checkpoint_path", required=True,  help="Path to model checpoint")
ap.add_argument("-ck_steps", "--chekpoint_steps", required=False,  nargs='+',   help="Epochs evaluated list of int. ")
ap.add_argument("-scorepathFM", "--score_output_pathFM", required=True, default='path/to/save/scores_fm_map',  help="Path where the anomaly maps from the feature maps will be saved.")
ap.add_argument("-scorepathIM", "--score_output_pathIM", required=True, default='path/to/save/scores_im_map',  help="Path where the anomaly maps from the images will be saved.")
ap.add_argument("-loss_df_path", "--loss_dataframe_path", required=True, default='path/to/save/loss_table', help="Path where the loss data frame will be saved.")
ap.add_argument("-reconsIM_Path", "--reconsIM_Path", required=True, default='path/to/reconstructed_images',   help="Path where the reconstructed images will be saved. ")
ap.add_argument("-reconsFM_Path", "--reconsFM_Path", required=True, default='path/to/reconstructed_feature_maps',   help="Path where the reconstructed feature maps will be saved. ")
ap.add_argument("-respath", "--res_output_path", required=True, default='/path/to/results_of_exp',   help="Path to the directory where the results will be saved. ")

args = vars(ap.parse_args())


class TestMvTec:
    def __init__(self, config =args):
        #  Model config
        self.img_size = 256
        self.batch_size = int(config['batch_size'])
        self.checkpoint_path = config['checkpoint_path']
        self.FMOnly =  config['FMOnly']
        ## Block D
        self.n_layer_vgg = config['n_layer_vgg']
        self.first_out_channel = config['n_first_out_channels_vgg']
        
        
        self.ssim_loss = pytorch_ssim.SSIM()
        
        # Data loader config 
        self.workers = int(config['workers'])
        self.MVTEC_object = config['MVTEC_object']
        self.data_root_directory =  config['data_root_directory']
        
        # Ouput config
        self.score_output_pathFM = config['score_output_pathFM']
        self.score_output_pathIM = config['score_output_pathIM']
        self.reconsIM_Path =  config['reconsIM_Path']
        self.reconsFM_Path =  config['reconsFM_Path']
        self.res_output_path = config['res_output_path']
        self.loss_path_df = config['loss_dataframe_path']


    def load_model(self, step):
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Load the model
        self.model =  ae( Init =  True,
                           n_layer_vgg = self.n_layer_vgg,
                           n_tail_layer_vgg = 2,
                           first_out_channel = self.first_out_channel,
                           FM =self.FMOnly,
                           num_classes = 2 )

        c_checkpoint_path = self.checkpoint_path.split('.')[0] + '_' + str(step) + '.pt'
        self.model.load_state_dict(torch.load(c_checkpoint_path), strict = False)
        self.model.eval()

        
    def data_loader(self):
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Test data loader
        test_transform = transforms.Compose([])
        test_transform.transforms.append(transforms.Resize((self.img_size,self.img_size)))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        
        test_dset = MVTECTestDataLoader(object_name=self.MVTEC_object, 
                                         root = self.data_root_directory,
                                        set_ = 'Test', 
                                        transform_process= test_transform)
        
        self.test_loader = torch.utils.data.DataLoader(
                test_dset,
                batch_size= self.batch_size,
                shuffle=False,
                num_workers= self.workers,
                pin_memory=False)
        
        train_dset = MVTECTestDataLoader(object_name=self.MVTEC_object, 
                                         root = self.data_root_directory,
                                         set_ = 'Train',
                                         transform_process= test_transform)
        
        self.train_loader = torch.utils.data.DataLoader(
                train_dset,
                batch_size= self.batch_size,
                shuffle=False,
                num_workers= self.workers,
                pin_memory=False)
        
        
        
    def inference(self, step, test_set = True, write_loss = False):
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Output directories
        c_score_output_pathFM = self.score_output_pathFM + '_' + str(step)
        c_score_output_pathIM = self.score_output_pathIM +  '_' + str(step)
        c_reconsIM_Path = self.reconsIM_Path + '_' + str(step)
        c_reconsFM_Path = self.reconsFM_Path + '_' + str(step)
        if test_set:
            loader = self.test_loader
            os.makedirs(c_score_output_pathFM, exist_ok=True)
            os.makedirs(c_score_output_pathIM, exist_ok=True)
            os.makedirs(c_reconsIM_Path, exist_ok=True)
            os.makedirs(c_reconsFM_Path, exist_ok=True)
        else:
            loader = self.train_loader
            c_score_output_pathFM  =  c_score_output_pathFM + '_Train'
            c_score_output_pathIM  =  c_score_output_pathIM + '_Train'
            os.makedirs(c_score_output_pathFM, exist_ok=True)
            os.makedirs(c_score_output_pathIM, exist_ok=True)
            
            
            c_reconsIM_Path  =  c_reconsIM_Path + '_Train'
            c_reconsFM_Path  =  c_reconsFM_Path + '_Train'
            os.makedirs(c_reconsIM_Path, exist_ok=True)
            os.makedirs(c_reconsFM_Path, exist_ok=True)
        ## Initi loss dataframe
        loss_cross_ent = torch.nn.CrossEntropyLoss()
        df = pd.DataFrame({'fname':[], 'Loss':[], 'Logit1':[], 'Logit2':[]})
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run inference
        with torch.no_grad():
            # Initialize the lists
            self.Losses = []
            self.cls =[]
            self.MSEFM = []
            self.MSEIM = []
            self.SSIMFM = []
            self.SSIMIM = []
            self.Labels = []
            self.Logit1 = []
            self.Logit2 = []
            self.exp_label = []
            l = []
            for c,(img, mask, label) in enumerate(loader):
                if self.FMOnly: # If no images reconstruction
                    logitImg, feature_map,  reconsFM  = self.model(img.cuda())
                else:
                    logitImg, feature_map, reconsImg, reconsFM  = self.model(img.cuda())
                feature_map = feature_map.cuda(0)
                reconsFM = reconsFM.cuda(0)
                loss1FM = F.mse_loss(reconsFM, feature_map)
                loss2FM = 1 - pytorch_ssim.ssim(feature_map, reconsFM)
                if not self.FMOnly:
                    reconsImg = reconsImg.cuda(0)
                    img = img.cuda(0)
                    loss1IM = F.mse_loss(img, reconsImg)
                    loss2IM = 1 - pytorch_ssim.ssim(img, reconsImg)
                #Loss calculations
                logitImg = logitImg.cuda(0)
                label_n = torch.zeros(self.batch_size)
                if label[0].split('/')[-2] != 'good':
                    self.exp_label.append(1)
                    label_n = torch.ones(1)
                if label[0].split('/')[-2] == 'good':
                    self.exp_label.append(0)
                label_n  = label_n.cuda(0)
                label_n = label_n.to(torch.long)
                loss_cls = loss_cross_ent(logitImg, label_n)
                feature_map = feature_map.permute(0,2,3,1)
                feature_map = feature_map.cpu().numpy()
                reconsFM = reconsFM.permute(0,2,3,1)
                reconsFM = reconsFM.cpu().numpy()
              
                if not self.FMOnly:
                    reconsImg = reconsImg.permute(0,2,3,1)
                    reconsImg = reconsImg.cpu().numpy()
                img = img.permute(0,2,3,1)
                img = img.cpu().numpy()

                if not self.FMOnly:
                    loss = 0.2*loss_cls + 0.2*loss1FM+0.2*loss2FM + 0.2*loss1IM+0.2*loss2IM
                else:
                    loss = 0.33*loss_cls + 0.33*loss1FM+0.33*loss2FM 
                
                # Append the resutls
                self.cls.append(loss_cls.item())
                self.MSEFM.append(loss1FM.item())
                self.SSIMFM.append(loss2FM.item())
                if not self.FMOnly:
                    self.MSEIM.append(loss1IM.item())
                    self.SSIMIM.append(loss2IM.item())
                else:
                    self.MSEIM.append('NA')
                    self.SSIMIM.append('NA')
                self.Losses.append(loss.item())
                self.Logit1.append(logitImg.cpu().numpy()[0,0])
                self.Logit2.append(logitImg.cpu().numpy()[0,1])
                self.Labels.append(label[0])

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # Calculate the anomaly maps
                anomaly_mapFM = self.scores_f(feature_map, reconsFM)
                if not self.FMOnly:
                    anomaly_mapIM = self.scores_f(img, reconsImg)
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # Write anomaly map
                l.append(label[0].split('/')[-2])
                os.makedirs(os.path.join(c_score_output_pathFM, label[0].split('/')[-2]), exist_ok=True)
                np.save(os.path.join(c_score_output_pathFM,
                                     label[0].split('/')[-2], label[0].split('/')[-1].split('.')[0] + '.npy'), anomaly_mapFM)
                
                os.makedirs(os.path.join(c_reconsFM_Path, label[0].split('/')[-2]), exist_ok=True)
                np.save(os.path.join(c_reconsFM_Path,
                                     label[0].split('/')[-2], label[0].split('/')[-1].split('.')[0] + '.npy'), reconsFM)
                if not self.FMOnly:
                    os.makedirs(os.path.join(c_score_output_pathIM, label[0].split('/')[-2]), exist_ok=True)
                    np.save(os.path.join(c_score_output_pathIM,
                                     label[0].split('/')[-2], label[0].split('/')[-1].split('.')[0] + '.npy'), anomaly_mapIM)
                    os.makedirs(os.path.join(c_reconsIM_Path, label[0].split('/')[-2]), exist_ok=True)
                    np.save(os.path.join(c_reconsIM_Path,
                                     label[0].split('/')[-2], label[0].split('/')[-1].split('.')[0] + '.npy'), reconsImg)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Write loss data frame
            if write_loss:
                df['fname'] = self.Labels
                df['exp_label'] = self.exp_label
                df['loss'] = self.Losses
                df['cls'] = self.cls
                df['MSEFM'] = self.MSEFM
                df['SSIMFM'] = self.SSIMFM
                df['MSEIM'] = self.MSEFM
                df['SSIMIM'] = self.SSIMFM
                df['Logit1'] = self.Logit1
                df['Logit2'] = self.Logit2
                loss_path_df = self.loss_path_df.split('.')[0] + '_' + str(step) + '.csv'
                df.to_csv(loss_path_df)

    def scores_f(self, img, reconstruction):
        """
        Args:
            img: image with size of (batch-size, channels, img_size_h, img_size_w)
        Returns:
            score map with shape (batch-size, channels, img_size_h, img_size_w)

        Note:
            batch-size = 1
        """
        scores = np.mean(((img - reconstruction) ** 2), axis=-1)
        return scores
    

    ###########################################
    # Main
    ###########################################
    def main(self, step):
        print(' 1 - Load the model \n')
        self.load_model(step)
        print(' 2 - Load the data \n')
        self.data_loader()
        print(' 3 - Infer the test set \n')
        self.inference(step, write_loss = True)
        print(' 4 - Infer the train set \n')
        self.inference(step, test_set = False, write_loss=False)

        

if __name__=="__main__":
    res_dir_full_path =os.path.join( args['data_root_directory'], args['MVTEC_object'], 'res')
    reconstruction_dir_full_path =os.path.join(args['res_output_path'])
    os.makedirs(res_dir_full_path, exist_ok=True)
    os.makedirs(reconstruction_dir_full_path, exist_ok=True)
    Test  = TestMvTec(args)
    # Evaluation of the model via the different checkpoints.
    for step  in args['chekpoint_steps']:
        print(f'~~~~~~~~~~~~~~~~~~~{step}~~~~~~~~~~~~~~~')
        Test.main(step)