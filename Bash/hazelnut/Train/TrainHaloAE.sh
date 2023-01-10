#!/bin/bash
conda deactivate
conda activate MIL_RNN_efficient
python ~/HaloAE/train.py --epochs 251  --model_name_checkpoint HaloAE_adapt_fixed_hazelnut.pt   --run_directory runs_HaloAE_adapt_fixed_hazelnut   --learning_rate 1e-4 --MVTEC_object hazelnut --data_root_directory /home/XXXX/LNENWork/MvTech --summury_path /home/XXXX/LNENWork/MvTech/TensorboardLog --weight_initialisation None --checkpoint_path /home/XXXX/LNENWork/MvTech/ModelsFeaturesMaps