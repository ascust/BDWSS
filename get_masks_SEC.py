import numpy as np
from PIL import Image
import os
import argparse
import cores.utils.misc as misc
import cores.utils.voc_cmap as voc_cmap
import cores.config as config
import time
import random
import datetime
import cPickle as pickle
import mxnet as mx
from cores.utils.CRF import CRF

def infer_use_SEC(image_folder, output_folder, image_list, net_symbol, weight_file, multi_label_file=None,
                  gpu=0, use_crf=False):
    misc.my_mkdir(output_folder)
    cmap = voc_cmap.get_cmap()

    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"]="0"
    mean_rgb = mx.nd.array(config.MEAN_RGB, ctx=mx.cpu()).reshape((1, 1, 3))

    crf_filter = CRF(scale_factor=1.0)

    if multi_label_file is not None:
        with open(multi_label_file, 'rb') as f:
            data_dict = pickle.load(f)

    seg_net = net_symbol.create_infer(config.CLASS_NUM, config.WORKSPACE)
    arg_dict, aux_dict, _ = misc.load_checkpoint(weight_file)
    mod = mx.mod.Module(seg_net, data_names=["data"], label_names=[], context=mx.gpu(gpu))
    mod.bind(data_shapes=[("data", (1, 3, config.INPUT_SIZE_SEC, config.INPUT_SIZE_SEC))],
             for_training=False, grad_req="null")

    initializer = mx.init.Normal()
    initializer.set_verbosity(True)
    mod.init_params(initializer=initializer, arg_params=arg_dict, aux_params=aux_dict, allow_missing=True)

    file_list = image_list
    random.shuffle(file_list)
    total = len(file_list)
    count = 0
    start_time = time.time()
    for im_path in file_list:
        count += 1
        if os.path.exists(os.path.join(output_folder, im_path.replace(".jpg", ".png"))):
            print "skipped %s because it already exists." % im_path
            continue

        #get mask
        buf = mx.nd.array(
            np.frombuffer(open(os.path.join(image_folder, im_path), 'rb').read(), dtype=np.uint8),
            dtype=np.uint8, ctx=mx.cpu())
        im = mx.image.imdecode(buf)
        h, w = im.shape[:2]
        resized_im = mx.image.imresize(im, config.INPUT_SIZE_SEC, config.INPUT_SIZE_SEC, interp=1)
        resized_im = resized_im.astype(np.float32)
        resized_im -= mean_rgb
        resized_im = mx.nd.transpose(resized_im, [2, 0, 1])
        resized_im = mx.nd.expand_dims(resized_im, 0)

        mod.forward(mx.io.DataBatch(data=[resized_im]))

        score = mx.nd.transpose(mod.get_outputs()[0].copyto(mx.cpu()), [0, 2, 3, 1])
        score = mx.nd.reshape(score, (score.shape[1], score.shape[2], score.shape[3]))
        up_score = mx.nd.transpose(mx.image.imresize(score, w, h, interp=1), [2, 0, 1])

        if multi_label_file is not None:
            tmp_label = data_dict[im_path.replace(".jpg", "")]
            image_level_labels = np.zeros((config.CLASS_NUM - 1))
            image_level_labels[tmp_label] = 1
            image_level_labels = np.insert(image_level_labels, 0, 1)
            image_level_labels = image_level_labels.reshape((config.CLASS_NUM, 1, 1))
            image_level_labels = mx.nd.array(image_level_labels, ctx=mx.cpu())

            up_score *= image_level_labels

        up_score = up_score.asnumpy()
        up_score[up_score < 0.00001] = 0.00001

        # #renormalize
        if use_crf:
            mask = np.argmax(crf_filter.inference(im.asnumpy(), np.log(up_score)), axis=0)
        else:
            mask = np.argmax(up_score, axis=0)

        out_img = np.uint8(mask)
        out_img = Image.fromarray(out_img)
        out_img.putpalette(cmap)
        out_img.save(os.path.join(output_folder, im_path.replace(".jpg", ".png")))
        elapsed_time = (time.time() - start_time)
        eta = int((total - count) * (elapsed_time/count))
        print "processed %s\t%d/%d\teta: %s" % (im_path, count, total, str(datetime.timedelta(seconds=eta)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--target", default= "initweb",
                        help="The target image folder. \"initweb\" is for web images with init-sec. "
                             "\"webweb\" is for web images with web-sec. "
                             "\"initvoc\" is for voc images with init-sec. default: initweb")
    parser.add_argument("--gpu", default=0, type=int,
                        help="Device indices. default: 0")
    args = parser.parse_args()

    misc.my_mkdir(config.CACHE_PATH)

    if args.target == "initweb":
        image_folder = config.WEB_IMAGE_FOLDER
        output_folder = os.path.join(config.CACHE_PATH, config.WEB_MASK_FOLDER_INITSEC)
        image_list = []
        for im_path in os.listdir(image_folder):
            if im_path.endswith(".jpg") and not im_path.startswith("."):
                image_list.append(im_path)

        model_name = "SEC_%s" % config.BASE_NET
        exec ("import cores.symbols." + model_name + " as net_symbol")
        weight_file = os.path.join(config.SNAPSHOT_FOLDER, "%s_init-1.params"%(model_name))
        infer_use_SEC(image_folder=image_folder, output_folder=output_folder, image_list=image_list,
                      net_symbol=net_symbol, weight_file=weight_file, gpu=args.gpu, use_crf=False)
    elif args.target == "webweb":
        image_folder = config.WEB_IMAGE_FOLDER
        output_folder = os.path.join(config.CACHE_PATH, config.WEB_MASK_FOLDER_WEBSEC)
        with open(os.path.join(config.CACHE_PATH, config.WEB_IMAGE_LABEL_FILE), "rb") as f:
            multi_label_dict = pickle.load(f)
            image_list = [i+".jpg" for i in multi_label_dict.keys()]
            model_name = "SEC_%s" % config.BASE_NET
        exec ("import cores.symbols." + model_name + " as net_symbol")
        weight_file = os.path.join(config.SNAPSHOT_FOLDER, "%s_web-1.params"%(model_name))
        infer_use_SEC(image_folder=image_folder, output_folder=output_folder, image_list=image_list,
                      net_symbol=net_symbol, weight_file= weight_file,
                      gpu=args.gpu, use_crf=True)
    elif args.target == "initvoc":
        image_folder = os.path.join(config.DATASET_PATH, config.VOC_TRAIN_IM_FOLDER)
        output_folder = os.path.join(config.CACHE_PATH, config.VOC_MASK_FOLDER_INITSEC)
        with open(os.path.join(config.DATASET_PATH, config.VOC_TRAIN_LIST)) as f:
            image_list = [i.strip()+".jpg" for i in f.readlines()]

        model_name = "SEC_%s" % config.BASE_NET
        exec ("import cores.symbols." + model_name + " as net_symbol")
        weight_file = os.path.join(config.SNAPSHOT_FOLDER, "%s_init-1.params"%(model_name))
        infer_use_SEC(image_folder=image_folder, output_folder=output_folder, image_list=image_list,
                      net_symbol=net_symbol, weight_file=weight_file,
                     multi_label_file=os.path.join(config.DATASET_PATH, config.VOC_TRAIN_MULTI_FILE),
                      gpu=args.gpu, use_crf=False)

        
    else:
        print "The target image folder. \"initweb\" is for web images with init-sec. "\
                "\"webweb\" is for web images with web-sec. "\
                "\"initvoc\" is for voc images with init-sec. default: initweb"
