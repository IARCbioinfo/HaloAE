# HaloAE
![Alt text](imgs/MainFigHaloNet.jpg?raw=true "HaloNetArchitecture")
##  Installation 

`conda env create --file env_HaloAE.yml`

`conda activate HaloAE`

## Dataset preparation 
For each MVTec class, the paths to the training and test images must be listed in text files named `train_images_list.txt` and `test_images_list.txt` respectively.  
To do this, the scrit `dataset/prepare_datasets.py` must be run such as:
`python dataset/prepare_datasets.py ---data_root_directory path/to/MVTec/dataset `

## Models traning
An example bash script to train the model is given in `Bash/hazelnut/Train/TrainHaloAE.sh`.
The Tensorboard log will be saved in `summury_path + run_directory` frolder.
The model checkpoints will be saved in the files `checkpoint_path + model_name_checkpoint` files.

**Warning :** The template will download a pre-trained VGG network, you may need to update the path in line 28 of the script:
`~/HaloAE/FeatureExtraction/vgg.py`


## Model testing
The test procedure consists of two steps: inference and scoring.

### Model inference
An example bashscript for inferring test images is given in `Bash/hazelnut/Test/Test_HaloAE_hazelnut.sh`.
The results are collected in `res_output_path`. At the inference stage, we save the following outputs for scoring:
+ The reconstructed images in `reconsIM_Path`,
+ The reconstructed feature map in `reconsFM_Path`,
+ The anomaly maps of the reconstructed images in `score_output_pathIM`,
+ The anomaly maps of the feature maps reconstructed in `score_output_path`,
+ The loss values in the `loss_dataframe_path` files.

Inference can be made for multiple checpoints listed by the `chekpoint_steps` argument.

### Scoring
Once the entire test dataset has been inferred, the ROC-AUC scores of the image-level detection and per-pixel segmentation can be computed via `scores.py`. An example bash script to generate these scores is given in `Bash/hazelnut/Test/Test_HaloAE_hazelnut.sh`.

**Warning:** The script is designed to evaluate 3 checkpoints at a time (see the `step_checkpoint` argument).

## Practical information
To download all the scripts at once, we create the `ToDownload` folder which should facilitate the use of `anonymous.4open.science`.


## Code architecture:
* train.py 
* inference.py
* scores.py
* HaloAE.py => Model
* dataset:
	+ curpaste.py => Cut&Paste data augmentation strategy
	+ MVTECDataLoader.py
	+ prepare_dataset.py 
* FeatureExtraction: => scripts for block B
	+ vgg19.py => Architecture of the vgg-19 network
	+ vgg.py => Load preprocessed Vgg 
	+ features_extractor.py => Creation of a multi-scale feature map
* encoder:
		+ HalonetEncoder.py => Halonet encoder see block C
* decoder:
	+ HalonetDecoder.py => Transposed Halonet (see block C) and transposed VGG (see block D)
* halonet_pytorch:
	+ __init__.py
	+ holonet_pytorch.py => local self-attention layer per block 
* pytorch_ssim:
	+ __init__.py => Script for SSIM operation
* Bash : => Organization of the proposed file 
	+ MVTec object:
		+ Train:
		+ Test

