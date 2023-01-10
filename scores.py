import torch
import numpy as np
import cv2
import os
import random
import pandas as pd
from skimage import measure
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
import warnings
import json
import torch
warnings.filterwarnings('ignore')
global ImgSize


import argparse



############################################################################
#  CONFIG
############################################################################

ap = argparse.ArgumentParser()
# ModelFMOn
ap.add_argument("-ims", "--ImgSize", required=False, default= 128,  type=int, help="Img size")
ap.add_argument("-obj", "--object", required=True, default= 'carpet',  type=str, help="Object name")
ap.add_argument("-root", "--rootdir", required=True, default= 'path/to/MvTech/',  type=str, help="MvTec rootdir")
ap.add_argument("-exp", "--exppath", required=True, default= 'ResFolder',  type=str, help="Path to exp folder.")
ap.add_argument("-st_chk", "--step_checkpoint", required=True,  nargs='+', help="List of checkpoint step.")
ap.add_argument("-FM", "--FMOnly", required=False, default= False,  type=bool, help="FMOnly = True not img")

args = vars(ap.parse_args())

Object =  args['object']
Anomaly = os.listdir(os.path.join(args['rootdir'], args['object'], 'test'))
RootDir = f"{args['rootdir']}/{Object}"
ExpFolder = args['exppath']
Steps = list(args['step_checkpoint'])
print(f'Epoch evaluated A = {Steps[0]}, B = {Steps[1]}, C = {Steps[2]}')
TestImgDir = f"{args['rootdir']}/{Object}/test"
ImgSize = args['ImgSize']
FMOnly = args['FMOnly']



############################################################################
# PATH
############################################################################
if not FMOnly:
    ScoresTrainIM_A = os.path.join(RootDir,ExpFolder,f'ScoresIM_{Steps[0]}_Train')
    ScoresTrainIM_B = os.path.join(RootDir,ExpFolder,f'ScoresIM_{Steps[1]}_Train')
    ScoresTrainIM_C = os.path.join(RootDir,ExpFolder,f'ScoresIM_{Steps[2]}_Train')

    ScoresTestIM_A = os.path.join(RootDir,ExpFolder,f'ScoresIM_{Steps[0]}')
    ScoresTestIM_B = os.path.join(RootDir,ExpFolder,f'ScoresIM_{Steps[1]}')
    ScoresTestIM_C = os.path.join(RootDir,ExpFolder,f'ScoresIM_{Steps[2]}')

ScoresTrainFM_A = os.path.join(RootDir,ExpFolder,f'ScoresFM_{Steps[0]}_Train')
ScoresTrainFM_B = os.path.join(RootDir,ExpFolder,f'ScoresFM_{Steps[1]}_Train')
ScoresTrainFM_C = os.path.join(RootDir,ExpFolder,f'ScoresFM_{Steps[2]}_Train')

ScoresTestFM_A = os.path.join(RootDir,ExpFolder,f'ScoresFM_{Steps[0]}')
ScoresTestFM_B = os.path.join(RootDir,ExpFolder,f'ScoresFM_{Steps[1]}')
ScoresTestFM_C = os.path.join(RootDir,ExpFolder,f'ScoresFM_{Steps[2]}')

if not FMOnly:
    ReconsTrainIM_A = os.path.join(RootDir,ExpFolder,f'ReconsIM_{Steps[0]}_Train')
    ReconsTrainIM_B = os.path.join(RootDir,ExpFolder,f'ReconsIM_{Steps[1]}_Train')
    ReconsTrainIM_C = os.path.join(RootDir,ExpFolder,f'ReconsIM_{Steps[2]}_Train')

    ReconsTestIM_A = os.path.join(RootDir,ExpFolder,f'ReconsIM_{Steps[0]}')
    ReconsTestIM_B = os.path.join(RootDir,ExpFolder,f'ReconsIM_{Steps[1]}')
    ReconsTestIM_C = os.path.join(RootDir,ExpFolder,f'ReconsIM_{Steps[2]}')


ReconsTrainFM_A = os.path.join(RootDir,ExpFolder,f'ReconsFM_{Steps[0]}_Train')
ReconsTrainFM_B = os.path.join(RootDir,ExpFolder,f'ReconsFM_{Steps[1]}_Train')
ReconsTrainFM_C = os.path.join(RootDir,ExpFolder,f'ReconsFM_{Steps[2]}_Train')

ReconsTestFM_A = os.path.join(RootDir,ExpFolder,f'ReconsFM_{Steps[0]}')
ReconsTestFM_B = os.path.join(RootDir,ExpFolder,f'ReconsFM_{Steps[1]}')
ReconsTestFM_C = os.path.join(RootDir,ExpFolder,f'ReconsFM_{Steps[2]}')

loss_path_file = os.path.join(RootDir,ExpFolder,'Loss.csv')
loss_path_file_A  = os.path.join(RootDir,ExpFolder,f'Loss_{Steps[0]}.csv')
loss_path_file_B  = os.path.join(RootDir,ExpFolder,f'Loss_{Steps[1]}.csv')
loss_path_file_C  = os.path.join(RootDir,ExpFolder,f'Loss_{Steps[2]}.csv')
############################################################################
# FONCTIONS
############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_fm_and_im(root_dir,class_anomaly,good = False):
    Reconstructions = []
    Labels = []
    Names = []
    if not good:
        for c in class_anomaly:
            imglist = os.listdir(os.path.join(root_dir,c))
            imglist
            imglist.sort()
            for i in imglist:
                Labels.append(c)
                Names.append(os.path.join(root_dir,c, i))
                im =  np.load(os.path.join(root_dir,c, i))
                im = np.squeeze(im)
                Reconstructions.append(np.mean(im, axis = -1))
    else:
        imglist = os.listdir(os.path.join(root_dir,'good'))
        imglist
        imglist.sort()
        for i in imglist:
            Labels.append('good')
            Names.append(os.path.join(root_dir, 'good', i))
            im =  np.load(os.path.join(root_dir, 'good',  i))
            im = np.squeeze(im)        
            Reconstructions.append(np.mean(im, axis = -1))
    Reconstructions = np.array(Reconstructions)
    return Reconstructions

def get_images(root_dir = TestImgDir, class_anomaly = Anomaly):
    Images = []
    Labels = []
    Names =  []
    for c in class_anomaly:
        imglist = os.listdir(os.path.join(root_dir,c))
        imglist.sort()
        for i in imglist:
            Names.append(os.path.join(root_dir,c, i))
            Labels.append(c)
            im = cv2.imread(os.path.join(root_dir,c, i))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (384,384))
            Images.append(im)
    Images = np.array(Images)
    return Images, Labels, Names

def get_masks(root_dir = f'{RootDir}/test',
              class_anomaly = Anomaly, size=(512,512)):
    Masks = []
    Labels = []
    Names = []
    for c in class_anomaly:
        imglist = os.listdir(os.path.join(root_dir,c))
        imglist.sort()
        for i in imglist:
            Labels.append(c)
            if c == 'good':
                # Careful with dimensions
                mask = np.zeros((size[0],size[1],3))
                Masks.append(mask)
                Names.append(os.path.join(root_dir,'good', i))
            else:
                path_img = os.path.join(root_dir,c, i)
                Names.append(os.path.join(root_dir,c, i))
                path_mask = path_img.replace('test','ground_truth' ).split('.')[0]+'_mask.png'
                mask = cv2.imread(path_mask)
                mask = cv2.resize(mask, (size[0],size[1]))
                Masks.append(mask)
    Masks = np.array(Masks)
    return Masks, Labels, Names

def get_scored_pictures(root_dir , class_anomaly = Anomaly, good =False, resized=False):
    Reconstructions = []
    Labels = []
    Names = []
    if not good:
        for c in class_anomaly:
            imglist = os.listdir(os.path.join(root_dir,c))
            imglist
            imglist.sort()
            for i in imglist:
                Labels.append(c)
                Names.append(os.path.join(root_dir,c, i))
                im =  np.load(os.path.join(root_dir,c, i))
                im = np.squeeze(im)
                if resized != False:
                    im =cv2.resize(im,(ImgSize,ImgSize), interpolation=cv2.INTER_LINEAR)
                Reconstructions.append(im)
    else:
        imglist = os.listdir(os.path.join(root_dir,'good'))
        imglist
        imglist.sort()
        for i in imglist:
            Labels.append('good')
            Names.append(os.path.join(root_dir, 'good', i))
            im =  np.load(os.path.join(root_dir, 'good',  i))
            im = np.squeeze(im)
            if resized != False:
                im = cv2.resize(im,(ImgSize,ImgSize), interpolation=cv2.INTER_LINEAR)
            Reconstructions.append(im)
    Reconstructions = np.array(Reconstructions)
    return Reconstructions, Labels, Names
    
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Post-Processing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def resize(img):
    img_r = []
    for i in range(img.shape[0]):
        img_r.append(cv2.resize(img[i], dsize=(ImgSize, ImgSize)))
    img_r = np.array(img_r)
    return img_r

def apply_gauss_filter(img, fsize=5):
    filteredimg = []
    for i in range(img.shape[0]):
        test = img[i,:,:]
        filteredimg.append(cv2.GaussianBlur(test,(fsize,fsize),0))
    filteredimg = np.array(filteredimg)
    filteredimg.shape
    return filteredimg

def Average_good_scores_map(NamesRecons, Scores, size=(512,512), ScoresGood = None):
    if not ScoresGood is None:
        good_tensor_mean = np.mean(ScoresGood, axis = 0)
    else:
        good_c = 0
        for e in NamesRecons:
            cat = e.split('/')[-2]
            if cat == 'good':
                good_c += 1

        good_tensor = np.zeros((good_c, size[0], size[1]))
        id_ = 0
        concat = 0
        for e in NamesRecons:
            cat = e.split('/')[-2]
            if cat == 'good':
                good_tensor[concat,:,:] = Scores[id_,:,:]
                concat += 1
            id_ += 1
        good_tensor_mean = np.mean(good_tensor, axis = 0)
    scores_minus_means = []
    for s in Scores:
        scores_minus_means.append(abs(s-good_tensor_mean))
    scores_minus_means =  np.array(scores_minus_means)
    return scores_minus_means


def get_minus_min(Scores_Halo,ScoresGood_Halo, ScoresIM_Halo, ScoresGoodIM_Halo ):
    scores_minus_means = Average_good_scores_map(NamesRecons, Scores_Halo, size=(ImgSize,ImgSize),   ScoresGood = ScoresGood_Halo)

    scoresIM_minus_means = Average_good_scores_map(NamesRecons, ScoresIM_Halo, size=(ImgSize,ImgSize), ScoresGood = ScoresGoodIM_Halo)

    return scores_minus_means, scoresIM_minus_means

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utils
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Binarization(mask, thres = 0., type = 0):
    if type == 0:
        mask = np.where(mask > thres, 1., 0.)
    elif type ==1:
        mask = np.where(mask > thres, mask, 0.)
    return mask  

def get_binary_labels(Labels, good=0):
    BLabels = []
    for l in Labels:
        if l == 'good':
            BLabels.append(good)
        else:
            
            BLabels.append(1-good)
    BLabels = np.array(BLabels)
    return BLabels

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Classification
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cls_prediction_image_level(LabelsT, BLabels, InvBLabels, Scores, process_name, ReturnDFs = False):
    # Prediction according to the mean
    predsM = np.mean(np.mean(Scores,axis=1),axis=1)    
    df_predMeans = pd.DataFrame(BLabels)
    df_predMeans['preds'] = predsM
    df_predMeans['Labels'] = LabelsT
    try:
        roc_auc_score_mean = round(roc_auc_score(df_predMeans.iloc[:,0], df_predMeans.iloc[:,1]), 4)
    except:
        # May fail if only one class
        roc_auc_score_mean = -1

    if roc_auc_score_mean < 0.5:
        df_predMeans = pd.DataFrame(InvBLabels)
        df_predMeans['preds'] = predsM
        df_predMeans['Labels'] = LabelsT
        try:
            roc_auc_score_mean = round(roc_auc_score(df_predMeans.iloc[:,0], df_predMeans.iloc[:,1]), 4)
        except:
            # May fail if only one class
            roc_auc_score_mean = -1

    # Prediction according to the max
    predsX = Scores.max(1).max(1)    # for detection
    df_predMaX = pd.DataFrame(BLabels)
    df_predMaX['preds'] = predsX
    df_predMaX['Labels'] = LabelsT
    try:
        roc_auc_scores_max = round(roc_auc_score(df_predMaX.iloc[:,0], df_predMaX.iloc[:,1]), 4)
    except:
        # May fail if only one class
        roc_auc_score_mean = -1
    if roc_auc_scores_max < 0.5 :
        df_predMaX = pd.DataFrame(InvBLabels)
        df_predMeans['preds'] = predsX
        df_predMeans['Labels'] = LabelsT
        try:
            roc_auc_score_mean = round(roc_auc_score(df_predMeans.iloc[:,0], df_predMeans.iloc[:,1]), 4)
        except:
            # May fail if only one class
            roc_auc_score_mean = -1

    if ReturnDFs:
        return df_predMeans, df_predMaX, roc_auc_score_mean, roc_auc_scores_max, process_name
    else: 
        return  roc_auc_score_mean, roc_auc_scores_max, process_name
        
def get_max_res_dict(dict_cls_res):
    best_by_mean = ['NA',0]
    best_by_max = ['NA',0]
    cls_by_mean = {}
    cls_by_max = {}
    NewKeyList = []
    for k in dict_cls_res.keys():
        NewKeyList.append(k[:-2])
        cls_by_mean[k] = dict_cls_res[k][0]
        cls_by_max[k] = dict_cls_res[k][1]
        if dict_cls_res[k][0] > best_by_mean[1]:
            best_by_mean[0] = k
            best_by_mean[1] = dict_cls_res[k][0]
        if dict_cls_res[k][1] > best_by_max[1]:
            best_by_max[0] = k
            best_by_max[1] = dict_cls_res[k][1]
    return best_by_mean, best_by_max

    
    
def extend_loss_table(DfLoss):
    anomaly = []
    exp_label = [] 
    predict_label = []
    for i in range(DfLoss.shape[0]):
        anomaly.append(DfLoss.iloc[i,1].split('/')[-2])
        if DfLoss.iloc[i,1].split('/')[-2] == 'good':
            exp_label.append(0)
        else:
            exp_label.append(1)
        predict_label.append(np.argmax(np.array(DfLoss.iloc[i,3:5].values)))
    DfLoss['anomaly'] = anomaly
    DfLoss['exp_label'] = exp_label
    DfLoss['predict_label'] = predict_label  
    return DfLoss

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Segmentation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_pixel_wise_seg_scores( maskH ,scores, size = 128, expect_fpr=0.3):
    # From https://github.com/YoungGod/DFR/blob/master/DFR-source/anoseg_dfr.py
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    masks = Binarization(maskH.sum(-1),0.)
    masks_resized = []
    for i in range(masks.shape[0]):
        masks_resized.append(cv2.resize(masks[i], dsize=(size, size)))
    masks_resized = np.array(masks_resized)
    masks = masks_resized
    # binary masks
    masks[masks <= 0.5] = 0
    masks[masks > 0.5] = 1
    masks = masks.astype(np.bool)

    Scores_resized = []
    for i in range(scores.shape[0]):
        Scores_resized.append(cv2.resize(scores[i], dsize=(size, size)))
    Scores_resized = np.array(Scores_resized)
    scores = Scores_resized
    
    labels = masks.any(axis=1).any(axis=1)
    #         preds = scores.mean(1).mean(1)
    preds_mean = scores.mean(1).mean(1)
    det_auc_score_mean = roc_auc_score(labels, preds_mean)
    det_pr_score = average_precision_score(labels, preds_mean)
    
    preds_max = scores.max(1).max(1)    # for detection
    det_auc_score_max = roc_auc_score(labels, preds_max)
    det_pr_score = average_precision_score(labels, preds_max)
    

    

    # auc score (per pixel level) for segmentation
    seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
    seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())

    return seg_pr_score, seg_auc_score, det_auc_score_mean, det_auc_score_max

def get_PRO_scores( masks, scores, expect_fpr=0.3, max_step=5000):
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score, average_precision_score
    from skimage import measure
    import pandas as pd
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())
    
    masks = Binarization(masks.sum(-1),0.)
    # binary masks
    masks[masks <= 0.5] = 0
    masks[masks > 0.5] = 1
    masks = masks.astype(np.bool)
    Scores_resized = []
    for i in range(scores.shape[0]):
        Scores_resized.append(cv2.resize(scores[i], dsize=(ImgSize, ImgSize)))
    Scores_resized = np.array(Scores_resized)
    scores = Scores_resized
    # auc score (image level) for detection
    labels = masks.any(axis=1).any(axis=1)
#         preds = scores.mean(1).mean(1)
    preds = scores.max(1).max(1)    # for detection
    det_auc_score = roc_auc_score(labels, preds)
    det_pr_score = average_precision_score(labels, preds)

    # auc score (per pixel level) for segmentation
    seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
    seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
    # metrics over all data
#     print(f"Det AUC: {det_auc_score:.4f}, Seg AUC: {seg_auc_score:.4f}")
#     print(f"Det PR: {det_pr_score:.4f}, Seg PR: {seg_pr_score:.4f}")

    # per region overlap and per image iou
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=np.bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []    # per region overlap
        iou = []    # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
        for i in range(len(binary_score_maps)):    # for i th image
            # pro (per region level)
            label_map = measure.label(masks[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image    # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], masks[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], masks[i]).astype(np.float32).sum()
            if masks[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
#             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~masks
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)

    # save results
    data = np.vstack([threds, fprs, pros_mean, pros_std, ious_mean, ious_std])
    df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                    'pros_mean', 'pros_std',
                                                    'ious_mean', 'ious_std'])


    # best per image iou
    best_miou = ious_mean.max()
#     print(f"Best IOU: {best_miou:.4f}")
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr    # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)    # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]    
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
#     print(fprs_selected.shape)
#     print('pros_mean_selected  ', pros_mean_selected)
#     print("pro auc ({}% FPR):".format(int(expect_fpr*100)), pro_auc_score)

    return df_metrics, pro_auc_score, pros_mean_selected, best_miou



############################################################################
# Upload data
############################################################################
print('\n\n  1- Data uploading')

Images, Labels, NamesImg =  get_images()
Masks_Halo, Labels, NamesMask = get_masks(size=(ImgSize,ImgSize))

if not FMOnly:
    ScoresTrainIM_A, LabelsG, NamesRecons = get_scored_pictures(ScoresTrainIM_A,   good = True, resized=True)
    ScoresTrainIM_B, LabelsG, NamesRecons = get_scored_pictures(ScoresTrainIM_B,  good = True, resized=True)
    ScoresTrainIM_C, LabelsG, NamesRecons = get_scored_pictures(ScoresTrainIM_C,  good = True, resized=True)


    ScoresTestIM_A, Labels, NamesRecons = get_scored_pictures(ScoresTestIM_A, resized=True)
    ScoresTestIM_B, Labels, NamesRecons = get_scored_pictures(ScoresTestIM_B, resized=True)
    ScoresTestIM_C, Labels, NamesRecons = get_scored_pictures(ScoresTestIM_C, resized=True)


    ReconsTrainIM_A = load_fm_and_im(ReconsTrainIM_A, Anomaly,good = True)
    ReconsTrainIM_B = load_fm_and_im(ReconsTrainIM_B,  Anomaly,good = True)
    ReconsTrainIM_C = load_fm_and_im(ReconsTrainIM_C,Anomaly,good = True)

    ReconsTestIM_A = load_fm_and_im(ReconsTestIM_A, Anomaly,good = False)
    ReconsTestIM_B = load_fm_and_im(ReconsTestIM_B,  Anomaly,good = False)
    ReconsTestIM_C = load_fm_and_im(ReconsTestIM_C, Anomaly,good = False)



ScoresTrainFM_A, LabelsG, NamesRecons = get_scored_pictures(ScoresTrainFM_A,  good = True)
ScoresTrainFM_B, LabelsG, NamesRecons = get_scored_pictures(ScoresTrainFM_B,  good = True)
ScoresTrainFM_C, LabelsG, NamesRecons = get_scored_pictures(ScoresTrainFM_C,  good = True)


ScoresTestFM_A, LabelsG, NamesRecons = get_scored_pictures(ScoresTestFM_A)
ScoresTestFM_B, LabelsG, NamesRecons = get_scored_pictures(ScoresTestFM_B)
ScoresTestFM_C, LabelsG, NamesRecons = get_scored_pictures(ScoresTestFM_C)

ReconsTrainFM_A = load_fm_and_im(ReconsTrainFM_A, Anomaly,good = True)
ReconsTrainFM_B = load_fm_and_im(ReconsTrainFM_B, Anomaly,good = True)
ReconsTrainFM_C = load_fm_and_im(ReconsTrainFM_C, Anomaly,good = True)

ReconsTestFM_A = load_fm_and_im(ReconsTestFM_A,  Anomaly,good = False)
ReconsTestFM_B = load_fm_and_im(ReconsTestFM_B,Anomaly,good = False)
ReconsTestFM_C = load_fm_and_im(ReconsTestFM_C, Anomaly,good = False)


############################################################################
# Post-Processing
############################################################################

print('\n\n 2- Data post-processing')
if not FMOnly:
    scoresIMFilter_A = apply_gauss_filter(ScoresTestIM_A)
    scoresIMFilter_B = apply_gauss_filter(ScoresTestIM_B)
    scoresIMFilter_C = apply_gauss_filter(ScoresTestIM_C)

    scoresFMNorm_A, scoresIMNorm_A,  = get_minus_min( ScoresTestFM_A ,ScoresTrainFM_A, ScoresTestIM_A, ScoresTrainIM_A )
    scoresFMNorm_B, scoresIMNorm_B,  = get_minus_min(ScoresTestFM_B ,ScoresTrainFM_B, ScoresTestIM_B, ScoresTrainIM_B )
    scoresFMNorm_C, scoresIMNorm_C,  = get_minus_min(ScoresTestFM_C ,ScoresTrainFM_C, ScoresTestIM_C, ScoresTrainIM_C )

else:
    scoresFMNorm_A = Average_good_scores_map(NamesRecons, ScoresTestFM_A, size=(ImgSize,ImgSize), ScoresGood = ScoresTrainFM_A)
    scoresFMNorm_B = Average_good_scores_map(NamesRecons, ScoresTestFM_B, size=(ImgSize,ImgSize), ScoresGood = ScoresTrainFM_B)
    scoresFMNorm_C = Average_good_scores_map(NamesRecons, ScoresTestFM_B, size=(ImgSize,ImgSize), ScoresGood = ScoresTrainFM_C)

scoresFMFilter_A = apply_gauss_filter(ScoresTestFM_A)
scoresFMFilter_B = apply_gauss_filter(ScoresTestFM_B)
scoresFMFilter_C = apply_gauss_filter(ScoresTestFM_C)

if not FMOnly:
    scoresIMNormFilter_A = apply_gauss_filter(scoresIMNorm_A,3)
    scoresIMNormFilter_B = apply_gauss_filter(scoresIMNorm_B,3)
    scoresIMNormFilter_C = apply_gauss_filter(scoresIMNorm_C,3)

scoresFMNormFilter_A = apply_gauss_filter(scoresFMNorm_A,3)
scoresFMNormFilter_B = apply_gauss_filter(scoresFMNorm_B,3)
scoresFMNormFilter_C = apply_gauss_filter(scoresFMNorm_C,3)


############################################################################
# Classification
############################################################################

print('\n\n  3- Anomaly detection at Image level')
print('\n\n  3.1-  According to matrices')
if not FMOnly:
    L_postprocess_matrix = [scoresIMNormFilter_A, scoresIMNormFilter_B, scoresIMNormFilter_C,
                            scoresIMFilter_A, scoresIMFilter_B, scoresIMFilter_C,
                            scoresIMNorm_A, scoresIMNorm_B, scoresIMNorm_C,
                            scoresFMNorm_A, scoresFMNorm_B, scoresFMNorm_C,
                            scoresFMFilter_A, scoresFMFilter_B, scoresFMFilter_C,
                            scoresFMNormFilter_A, scoresFMNormFilter_B, scoresFMNormFilter_C]


    L_ProcessName = [       'scoresIMNormFilter_A', 'scoresIMNormFilter_B', 'scoresIMNormFilter_C',
                            'scoresIMFilter_A', 'scoresIMFilter_B', 'scoresIMFilter_C',
                            'scoresIMNorm_A', 'scoresIMNorm_B', 'scoresIMNorm_C',
                            'scoresFMNorm_A', 'scoresFMNorm_B', 'scoresFMNorm_C',
                            'scoresFMFilter_A', 'scoresFMFilter_B', 'scoresFMFilter_C',
                            'scoresFMNormFilter_A', 'scoresFMNormFilter_B', 'scoresFMNormFilter_C']
else:
    L_postprocess_matrix = [scoresFMNorm_A, scoresFMNorm_B, scoresFMNorm_C,
                            scoresFMFilter_A, scoresFMFilter_B, scoresFMFilter_C,
                            scoresFMNormFilter_A, scoresFMNormFilter_B, scoresFMNormFilter_C]


    L_ProcessName = ['scoresFMNorm_A', 'scoresFMNorm_B', 'scoresFMNorm_C',
                     'scoresFMFilter_A', 'scoresFMFilter_B', 'scoresFMFilter_C',
                     'scoresFMNormFilter_A', 'scoresFMNormFilter_B', 'scoresFMNormFilter_C']

BLabels = get_binary_labels(Labels) # If Good Blabel = 0, Else Blabel = 1
InvBLabels = get_binary_labels(Labels, good=1) # If Good Blabel = 1, Else Blabel = 0


dict_cls_res = {}
for i in range(len(L_postprocess_matrix)):
    roc_auc_score_mean, roc_auc_scores_max, process_name =  cls_prediction_image_level(Labels, BLabels, InvBLabels, L_postprocess_matrix[i],  L_ProcessName[i])
    process_name_mean = process_name.split('_')[0] + '_mean_' + process_name.split('_')[1] 
    dict_cls_res[process_name_mean] = roc_auc_score_mean 
    process_name_max = process_name.split('_')[0] + '_max_' + process_name.split('_')[1] 
    dict_cls_res[process_name_max] =  roc_auc_scores_max


print('\n\n  3.1-  According to loss')

def get_loss_anomaly_detection_scores(DfLoss, FMOnly):
    DfLoss = extend_loss_table(DfLoss)
    
    DfLoss['Inv_label'] = [-1] * DfLoss.shape[0]
    DfLoss.loc[DfLoss['exp_label'] == 0,'Inv_label'] = 1
    DfLoss.loc[DfLoss['exp_label'] == 1,'Inv_label'] = 0
    
    TotlossROC = round(roc_auc_score(DfLoss['exp_label'], DfLoss['loss']), 4)
    if TotlossROC < 0.5:
        TotlossROC = round(roc_auc_score(DfLoss['Inv_label'], DfLoss['loss']), 4)
        
    ClsLossScores = round(roc_auc_score(DfLoss['exp_label'], DfLoss['cls']), 4)
    if ClsLossScores < 0.5:
        ClsLossScores = round(roc_auc_score(DfLoss['Inv_label'], DfLoss['cls']), 4)
        
    MSEFMLossScores = round(roc_auc_score(DfLoss['exp_label'], DfLoss['MSEFM']), 4)
    if MSEFMLossScores < 0.5:
        MSEFMLossScores = round(roc_auc_score(DfLoss['Inv_label'], DfLoss['MSEFM']), 4)
        
    SSIMFMLossScores = round(roc_auc_score(DfLoss['exp_label'], DfLoss['SSIMFM']), 4)
    if SSIMFMLossScores < 0.5:
        SSIMFMLossScores = round(roc_auc_score(DfLoss['Inv_label'], DfLoss['SSIMFM']), 4)
   
    if not FMOnly:
        MSEIMLossScores = round(roc_auc_score(DfLoss['exp_label'], DfLoss['MSEIM']), 4)
        if MSEIMLossScores < 0.5:
            MSEIMLossScores = round(roc_auc_score(DfLoss['Inv_label'], DfLoss['MSEIM']), 4)
   
        SSIMIMLossScores = round(roc_auc_score(DfLoss['exp_label'], DfLoss['SSIMIM']), 4)
        if SSIMIMLossScores < 0.5:
            SSIMIMLossScores = round(roc_auc_score(DfLoss['Inv_label'], DfLoss['SSIMIM']), 4)
   
    else:
        MSEIMLossScores = 'NA'
        SSIMIMLossScores = 'NA'
    return TotlossROC, ClsLossScores, MSEFMLossScores, SSIMFMLossScores, MSEIMLossScores, SSIMIMLossScores

try:
    DfLoss = pd.read_csv(loss_path_file)
    TotlossROC, ClsLossScores, MSEFMLossScores, SSIMFMLossScores, MSEIMLossScores, SSIMIMLossScores = get_loss_anomaly_detection_scores(
DfLoss, FMOnly)
    dict_cls_res['TotlossROC'] = TotlossROC
    dict_cls_res['ClsLossScores'] = ClsLossScores
    dict_cls_res['MSEFMLossScores'] = MSEFMLossScores
    dict_cls_res['SSIMFMLossScores'] = SSIMFMLossScores
    dict_cls_res['MSEIMLossScores'] = MSEIMLossScores
    dict_cls_res['SSIMIMLossScores'] = SSIMIMLossScores
    # print('Anomaly detction at image level dict (dict_cls_res ) : \n ', dict_cls_res, '\n\n ')
 
    
except:
    DfLoss_A = pd.read_csv(loss_path_file_A)
    TotlossROC_A, ClsLossScores_A, MSEFMLossScores_A, SSIMFMLossScores_A, MSEIMLossScores_A, SSIMIMLossScores_A = get_loss_anomaly_detection_scores(
DfLoss_A, FMOnly)
    dict_cls_res['TotlossROC_A'] = TotlossROC_A
    dict_cls_res['ClsLossScores_A'] = ClsLossScores_A
    dict_cls_res['MSEFMLossScores_A'] = MSEFMLossScores_A
    dict_cls_res['SSIMFMLossScores_A'] = SSIMFMLossScores_A
    dict_cls_res['MSEIMLossScores_A'] = MSEIMLossScores_A
    dict_cls_res['SSIMIMLossScores_A'] = SSIMIMLossScores_A
        
    DfLoss_B = pd.read_csv(loss_path_file_B)
    TotlossROC_B, ClsLossScores_B, MSEFMLossScores_B, SSIMFMLossScores_B, MSEIMLossScores_B, SSIMIMLossScores_B = get_loss_anomaly_detection_scores(
DfLoss_B, FMOnly)
    dict_cls_res['TotlossROC_B'] = TotlossROC_B
    dict_cls_res['ClsLossScores_B'] = ClsLossScores_B
    dict_cls_res['MSEFMLossScores_B'] = MSEFMLossScores_B
    dict_cls_res['SSIMFMLossScores_B'] = SSIMFMLossScores_B
    dict_cls_res['MSEIMLossScores_B'] = MSEIMLossScores_B
    dict_cls_res['SSIMIMLossScores_B'] = SSIMIMLossScores_B
    
    DfLoss_C = pd.read_csv(loss_path_file_C)
    TotlossROC_C, ClsLossScores_C, MSEFMLossScores_C, SSIMFMLossScores_C, MSEIMLossScores_C, SSIMIMLossScores_C = get_loss_anomaly_detection_scores(
DfLoss_C, FMOnly)
    dict_cls_res['TotlossROC_C'] = TotlossROC_C
    dict_cls_res['ClsLossScores_C'] = ClsLossScores_C
    dict_cls_res['MSEFMLossScores_C'] = MSEFMLossScores_C
    dict_cls_res['SSIMFMLossScores_C'] = SSIMFMLossScores_C
    dict_cls_res['MSEIMLossScores_C'] = MSEIMLossScores_C
    dict_cls_res['SSIMIMLossScores_C'] = SSIMIMLossScores_C
    


############################################################################
# Segmentation
############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pixel-wise (ROCAUC)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\n\n  4 - Anomaly localization (pixel-wise)')

dict_seg_pixelwise_res = {}
dict_pr_pixelwise_res = {}
dict_cls_max_res = {}
dict_cls_mean_res = {}

for i in range(len(L_postprocess_matrix)):
    seg_pr_score, seg_auc_score, det_auc_score_mean, det_auc_score_max = get_pixel_wise_seg_scores( Masks_Halo,
                                                                                                   L_postprocess_matrix[i])
    dict_pr_pixelwise_res[L_ProcessName[i]] =  seg_pr_score
    dict_seg_pixelwise_res[L_ProcessName[i]] =  seg_auc_score
    process_name = L_ProcessName[i]
    process_name_mean = 'DFR' + process_name.split('_')[0] + '_mean_'+ process_name.split('_')[1]
    process_name_max = 'DFR' + process_name.split('_')[0] + '_max_'+ process_name.split('_')[1]
    dict_cls_res[process_name_mean] = det_auc_score_mean
    dict_cls_res[process_name_max] = det_auc_score_max


SegScoresAUC = {}
SegScoresAUC = dict(sorted(dict_seg_pixelwise_res.items(), key=lambda item: item[1], reverse=True))




if not FMOnly:
    dict_cls_res = dict(sorted(dict_cls_res.items(), key=lambda item: item[1], reverse=True))


# TO UNCONMMENT TO GET THE IMAGE LEVEL DETECTION SCORES FOR ALL OUTPUTS
# print('\n Anamoly detection at image-level ROC-AUC scores : \n ', dict_cls_res)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Per-region (PRO ROC AUC)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# if not FMOnly:
#     process_name_matrices = {'scoresIMNormFilter_A':scoresIMNormFilter_A, 'scoresIMNormFilter_B':scoresIMNormFilter_B, 
#                              'scoresIMNormFilter_C':scoresIMNormFilter_C, 'scoresIMFilter_A':scoresIMFilter_A, 
#                              'scoresIMFilter_B':scoresIMFilter_B, 'scoresIMFilter_C':scoresIMFilter_C,
#                              'scoresIMNorm_A':scoresIMNorm_A, 'scoresIMNorm_B':scoresIMNorm_B, 
#                              'scoresIMNorm_C':scoresIMNorm_C, 'scoresFMNorm_A':scoresFMNorm_A, 
#                              'scoresFMNorm_B':scoresFMNorm_B , 'scoresFMNorm_C':scoresFMNorm_C ,
#                              'scoresFMFilter_A':scoresFMFilter_A, 'scoresFMFilter_B':scoresFMFilter_B,
#                              'scoresFMFilter_C':scoresFMFilter_C,'scoresFMNormFilter_A': scoresFMNormFilter_A, 
#                              'scoresFMNormFilter_B':scoresFMNormFilter_B, 'scoresFMNormFilter_C':scoresFMNormFilter_C  }
# else:
#     process_name_matrices = {'scoresFMNorm_A':scoresFMNorm_A, 
#                              'scoresFMNorm_B':scoresFMNorm_B , 'scoresFMNorm_C':scoresFMNorm_C ,
#                              'scoresFMFilter_A':scoresFMFilter_A, 'scoresFMFilter_B':scoresFMFilter_B,
#                              'scoresFMFilter_C':scoresFMFilter_C,'scoresFMNormFilter_A': scoresFMNormFilter_A, 
#                              'scoresFMNormFilter_B':scoresFMNormFilter_B, 'scoresFMNormFilter_C':scoresFMNormFilter_C  }
    
# top_best_auc_matrices =[ 'scoresFMNormFilter_C']
# dict_seg_pro_res = {}
# for pn in top_best_auc_matrices:
#     df_metrics, pro_auc_score, pros_mean_selected, best_miou = get_PRO_scores(Masks_Halo, process_name_matrices[pn]) 
#     dict_seg_pro_res[pn] = pro_auc_score
    
    

print('\n\n----------------------------------------------------------------------------------------- \n ')
print(f"Image-level detection ROC-AUC score in percent for {Object}:  ", round(dict_cls_res['scoresFMNormFilter_mean_C'] * 100,2), '\n')

print(f"Pixel-wise ROC-AUC score in percent for {Object}:  ", round(SegScoresAUC['scoresFMNormFilter_C'] * 100,2))

