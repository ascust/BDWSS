BASE_NET = "vgg16" #vgg16 or resnet50
WEB_IMAGE_FOLDER = "dataset/web_images"
CACHE_PATH = "cache"

#for dataset
#SBD_PATH is the one named "dataset", which has "cls", "img", "inst", "train.txt" and "val.txt".
#please download at https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
#VOCDEVKIT_PATH is the one named "VOC2012", which has "Annotations", "ImageSets", "JPEGImages", "SegmentationClass",
DATASET_PATH = "dataset"
SBD_PATH = "dataset/benchmark_RELEASE/dataset"
VOCDEVKIT_PATH = "dataset/VOCdevkit/VOC2012"
VOC_TRAIN_MULTI_FILE = "voc_multi_file.p"
VOC_TRAIN_IM_FOLDER = "train_images"
VOC_VAL_IM_FOLDER = "val_images"
VOC_VAL_MASK_FOLDER = "val_masks"
VOC_TRAIN_LIST = "train_list.txt"
VOC_VAL_LIST = "val_list.txt"

LOG_FOLDER = "log"
SNAPSHOT_FOLDER = "snapshots"
OUTPUT_FOLDER = "outputs"

CLASS_NUM = 21 # in this case voc dataset.


#parameters for filter images
FG_TH_CURCLASS_LOW = 0.1 # the current class pixels at least take 30% of the whole image
FG_TH_CURCLASS_HI = 0.8 # the current class pixels should not exceed 70% of the whole image
FG_TH_OTHER_HI = 0.3 # other class pixels cannot exceed 10%
MAX_PER_CLASS = 4000 #maximum images per class

WEB_MASK_FOLDER_INITSEC = "web_masks_initsec"
WEB_MASK_FOLDER_WEBSEC = "web_masks_websec"
VOC_MASK_FOLDER_INITSEC = "voc_masks_initsec"
TMP_GC_RESULTS_FOLDER = "tmp_gc_results"
FINAL_WEB_MASK_FOLDER = "final_web_masks"
FINAL_VOC_MASK_FOLDER = "final_voc_masks"


WEB_IMAGE_FLIST = "web_train_list.txt"
WEB_IMAGE_LABEL_FILE = "web_im_multilabel_file.p"

MEAN_RGB = (123, 117, 104) #RGB not BGR

#training params for FG BG cue networks.
EPOCH_SIZE_FG_INITSEC = 8000
BATCH_SIZE_FG_INITSEC = 15
LR_FG_INITSEC = 1e-3
LR_DECAY_FG_INITSEC = 2000
FG_CUE_FILE_INITSEC = "fg_cue_initsec.p"

EPOCH_SIZE_FG_WEBSEC = 8000
BATCH_SIZE_FG_WEBSEC = 30
LR_FG_WEBSEC = 1e-3
LR_DECAY_FG_WEBSEC = 2000
FG_CUE_FILE_WEBSEC = "fg_cue_websec.p"

SALIENCY_TH_FG = 0.2

LR_BG_INITSEC = 1e-3
EPOCH_SIZE_BG_INITSEC = 8000
BATCH_SIZE_BG_INITSEC = 15
LD_DECAY_BG_INITSEC = 2000
BG_CUE_FILE_INISEC = "bg_cue_initsec.p"

LR_BG_WEBSEC = 1e-3
EPOCH_SIZE_BG_WEBSEC = 8000
BATCH_SIZE_BG_WEBSEC = 30
LR_DECAY_BG_WEBSEC = 2000
BG_CUE_FILE_WEBSEC = "bg_cue_websec.p"

SALIENCY_TH_BG = 0.1

#training params for SEC models
CUE_FILE_INITSEC = "cue_initsec.p"
INPUT_SIZE_SEC = 320
DOWN_SAMPLE_SEC = 8 #network resolution
Q_FG = 0.996
Q_BG = 0.999


LR_INITSEC = 1e-3
LR_DECAY_INITSEC = 2000
EPOCH_SIZE_INITSEC = 8000
BATCH_SIZE_INITSEC = 15

CUE_FILE_WEBSEC = "cue_websec.p"
LR_WEBSEC = 1e-4
LR_DECAY_WEBSEC = 2000
EPOCH_SIZE_WEBSEC = 8000
BATCH_SIZE_WEBSEC = 15



WD = 5e-4
MOMENTUM = 0.9
MEM_MIRROR = False #useful for small gpu memory when sets True
WORKSPACE = 512


#about grabcut refine
MAX_SAMPLE_GC = 100
MAX_TRIAL_GC = 500
OFFSET_GC = 0.05 #bounding box jittering. "left" and "right" may have offset of +-OFFSET*w. "top" and "bottom" may have offset +-OFFSET*h
#if the foreround has less pixles than MIN_DIM_TH*w*h*MIN_DIM_TH, then it is invalid.
#also the bounding box can not be smaller than MIN_DIM_TH*w and h*MIN_DIM_TH correspondingly
MIN_DIM_TH_GC = 0.2
MARGIN_GC = 4 #assume the object is at least 8 pixels from the sides
FG_TH_GC = 0.7 # pixels with larger values than 70% of max will be set to forground
USE_IGNORE_GC = True #whether set the inconfident pixels to ignore, which is 255.

#train web model
LR_WM = 16e-4
LR_DECAY_WM = 7000
EPOCH_SIZE_WM = 1000
MAX_EPOCH_WM = 20
BATCH_SIZE_WM = 16
CROP_SIZE_WM = 320
SCALE_RANGE_WM = [0.7, 1.3]
DOWN_SAMPLE_VALUE_WM = 8


#train final model
LR_FM = 16e-4
LR_DECAY_FM = 7000
EPOCH_SIZE_FM = 700
MAX_EPOCH_FM = 30
BATCH_SIZE_FM = 16
CROP_SIZE_FM = 320
SCALE_RANGE_FM = [0.7, 1.3]
DOWN_SAMPLE_VALUE_FM = 8

#for evaluate init and final models
MAX_INPUT_DIM = 800
MIN_PIXEL_TH = 100
CPU_WORKER_NUM = 8

EVAL_WAIT_TIME = 0.3 # hour
EVAL_SCALE_LIST = [0.7, 1.0, 1.3] #multi scale evaluation can be used as [0.7, 1.0, 1.3]
INTERP_METHOD = 1# 1 for bilinear and 2 for bicubic

CRF_POS_XY_STD = 2
CRF_POS_W = 3
CRF_BI_RGB_STD = 3
CRF_BI_XY_STD = 55
CRF_BI_W = 4

