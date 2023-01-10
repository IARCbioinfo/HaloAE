# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from collections import OrderedDict
from itertools import chain
import torch.utils.data as data
import random
from PIL import Image
from skimage.transform import resize
import cv2
random.seed(123)



class Repeat(data.Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]

class MVTECTrainingDataLoader(data.Dataset):
    def __init__(self, object_name ="hazelnut", 
                 root='/path/to/MvTech',
                transform_process= None):
        
        # Get iamge list
        self.img_path_list_file = f'{root}/{object_name}/train_images_list.txt'
        with open(self.img_path_list_file, 'r') as f:
            content =  f.readlines()
        self.files_list = []
        for x in content:
            x =  x.strip()
            self.files_list.append(x)
            
        self.transform = transform_process
    def __getitem__(self,index):
        imgname = self.files_list[index] 
        img =  Image.open(self.files_list[index])
        if "zipper" in imgname or "screw" in imgname or "grid" in imgname:
            img = np.expand_dims(np.array(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            
        if self.transform is not None:
            img_t= self.transform(img)
        return img_t        
    def __len__(self):
        return len(self.files_list)

class MVTECTestDataLoader(data.Dataset):
    def __init__(self, object_name ="carpet", 
                 root='/gpfsscratch/rech/ohv/ueu39kt/MvTech', 
                 set_='Test',
                transform_process= None):

        self.img_size = 256
        self.img_path_list_file = f'{root}/{object_name}/test_images_list.txt'
        with open(self.img_path_list_file, 'r') as f:
            content =  f.readlines()
        self.files_list = []
        for x in content:
            x =  x.strip()
            self.files_list.append(x)
        self.transform = transform_process

    def __getitem__(self,index):
        imgname = self.files_list[index] 
        assert self.img_size%2==0, 'Img size should be even'
        img =  Image.open(self.files_list[index])
        if "zipper" in imgname or "screw" in imgname or "grid" in imgname:
            img = np.expand_dims(np.array(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        if self.transform is not None:
            img_t= self.transform(img)
           
        if imgname.split('/')[-2] == "good":
            mask = np.zeros((self.img_size, self.img_size))
        else:
            mask = np.zeros((self.img_size, self.img_size))
            mask_path = imgname.replace('test', 'ground_truth' ).split('.')[0]+ "_mask.png"
            mask =  cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask,(self.img_size, self.img_size))

        return img_t, mask, imgname
        

    def __len__(self):
        return len(self.files_list)


if __name__ == "__main__":
    batch_size = 1
    trainds = MVTECTrainingDataLoader()
    train_loader  = torch.utils.data.DataLoader(trainds, batch_size=batch_size, shuffle=True)
    for i, img in enumerate(train_loader):
        print(i, img)
        break
