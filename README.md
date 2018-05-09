# DENPENDENCIES
Our code is implemented in MXNet. The SEC model used in our framework (Seed, Expand, Constrain: Three Principles for Weakly-Supervised Image Segmentation), which was originally implemented in Caffe([Original Github](https://github.com/kolesman/SEC)), is also reimplemented in MXNet. 

Please refer to the official website of MXNet ([HERE](https://mxnet.apache.org)) for installation. Also make sure MXNet is compiled with OpenCV support.
The other dependent python packages can be found in "dependencies.txt". Please run:

```pip install -r dependencies.txt```

# DATASET
There will be three datasets involved, PASCAL VOC12([HERE](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)), SBD([HERE](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tg)) and Web images([HERE](https://cloudstor.aarnet.edu.au/plus/s/SPxXONHjYjFbbny/download)).
Extract them and put them into folder "dataset", and then run:

```python create_dataset.py```

# TRAINING
1. Download Models
Please download pretrained models ([HERE](https://1drv.ms/u/s!ArsE1Wwv6I6dgQGqn_nDGobaSSSf)), which includes vgg16 and resnet50 pretrained on Imagenet. Please extract the file and put the files into folder "models".
2. Start Training
In cores/config.py, all the parameters are shown. The most important one "BASE_NET", which defines the backbone of the model. Choose between "vgg16" and "resnet50".
Please follow "pipeline.sh" to run the programs. It is worth noting that most of the scripts can be executed multiple times to speed things up. Refer to "pipeline.sh" for more details.

# EVALUATION
Download the trained models, Resnet50 ([HERE](https://1drv.ms/u/s!ArsE1Wwv6I6dfsCommqv8dQLU9s)) or VGG16 ([HERE](https://1drv.ms/u/s!ArsE1Wwv6I6df_zvihgMmDbRfNw)), and put it in the folder "snapshots".
In cores/config.py, set "BASE_NET" to "vgg16" or "resnet50" to choose the desired model, and run:

```python eval_seg_model.py --model final --gpu 0 --epoch 19```

There are other flags:

```--savemask``` is used to save masks, outputs will be saved in "outputs" folder.

```--crf``` is used to use CRF as postprocessing

# TRAINING ON YOUR OWN WEB IMAGES
The provided dataset of web images include 76,683 web images searched from Bing. You can also try using your own images as long as it is consistent with our naming convention. 

The images should be named as:

```ID_XXXXX.jpg```

"ID" is the class ID in VOC. Since the background is "0", a valid "ID" is {1,2 ..., 20} in the case of PASCAL VOC. "XXXXX" can be anything. 