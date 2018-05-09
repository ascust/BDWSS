#!/bin/bash
#pipe line for the whole program. Some scripts can be executed multiple times for speeding things up.

#train init-sec
python train_fg_cues.py --gpus 0,1,2,3 --model init
python train_bg_cues.py --gpus 0,1,2,3 --model init
python train_SEC.py --gpus 0,1,2,3 --model init


#get initial masks for web images from SEC model. This can be executed multiple times with the same gpu of different gpu.
python get_masks_SEC.py --gpu 0 --target initweb
python filter_images.py #filter images from a large collection of web images.

#train web-sec
python train_fg_cues.py --gpus 0,1,2,3 --model web
python train_bg_cues.py --gpus 0,1,2,3 --model web
python train_SEC.py --gpus 0,1,2,3 --model web

#get masks for web images for retrained model. This can be executed multiple times with the same or different gpu.
python get_masks_SEC.py --gpu 0 --target webweb

#grabcut refinement. This can be executed multiple times.
#It is recommended to use a cluster to get the results in shorter time.
#This step would be very slow without a cluster due to multiple grabcut iterations.
#To skip this module, replace the two lines with "python generate_gc_masks.py --nogc"
python gc_refine.py
#generate masks for grab cut refinement
python generate_web_masks.py


#train and evaluate Web-FCN model. eval_loop.py can be executed during the training so that when there is a new snapshot,
# the evaluation will be conducted. Find the best model through this. Please also note that when a snapshot is evaluated,
# a folder will be created in "outputs" folder. eval_loop.py will determine whether a snapshot has been evaluated based on this.
# So for a freshing start, please delete the "outputs" folder. The evaluation results will also be stored in a log file
# called "eval_model.log" in folder "log".
python train_seg_model.py --model web --gpus 0,1,2,3
python eval_loop.py --model web --gpu 0

#get estimated masks for voc training images using the best model in the previous course. X is the epoch number
python get_masks_SEC.py --gpu 0 --target initvoc
python est_voc_train_masks.py --epoch X --gpu 0

#train and evaluate final model
python train_seg_model.py --model final --gpus 0,1,2,3
python eval_loop.py --model final --gpu 0
