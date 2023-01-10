import os
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-datadir", "--data_root_directory", required=True, default= 'path/to/MVTEC/object/img/folder',   help="Root directory of the dataset.")
args = vars(ap.parse_args())
MvTec_Rootdir = args['data_root_directory']

objects = ['carpet', 'grid', 'leather', 'tile', 'wood',
			'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
			'pill', 'screw', 'toothbrush', 'transistor','zipper']

for obj in objects:
	train_imgs_path = os.path.join(MvTec_Rootdir, obj, 'train', 'good')
	train_imgs_list = os.listdir(train_imgs_path)
	with open(os.path.join(MvTec_Rootdir, obj, 'train_images_list.txt'), 'w+') as f:
		for im_tr in os.listdir(train_imgs_path):
				f.write(os.path.join(MvTec_Rootdir, obj, 'train', 'good',im_tr ) + '\n')
	test_folder = os.path.join(MvTec_Rootdir, obj, 'test')
	anomaly_list = os.listdir(test_folder)
	with open(os.path.join(MvTec_Rootdir, obj, 'test_images_list.txt'), 'w+') as ft:
		for anomaly in anomaly_list:
			test_imgs_list = os.listdir(os.path.join(test_folder, anomaly ))
			for im_te in test_imgs_list:
				ft.write(os.path.join(MvTec_Rootdir, obj, 'test', anomaly ,im_te ) + '\n')
	
			