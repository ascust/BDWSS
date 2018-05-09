import argparse
from cores.seg_wrapper import train_seg_wrapper
import os
import mxnet as mx
import logging
import cores.config as config
import cores.utils.misc as misc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpus", default="0",
                        help="Device indices.")
    parser.add_argument("--epoch", default=0, type=int,
                        help="Starting epoch.")
    parser.add_argument("--lr", default=-1, type=float,
                        help="Learning rate.")
    parser.add_argument("--model", default="web",
                        help="train the web model or final model. either \"web\" or \"final\"")

    args = parser.parse_args()
    misc.my_mkdir(config.SNAPSHOT_FOLDER)
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = str(config.CPU_WORKER_NUM)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    assert args.model in ["web", "final"], "worng model type. Should be either \"web\" or \"final\""
    log_file_name = os.path.join(config.LOG_FOLDER, "train_%s_model.log"%args.model)
    if os.path.exists(log_file_name) and args.epoch==0:
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)


    if args.model == "web":
        model_name = "web_fcn_%s" % config.BASE_NET
        exec ("import cores.symbols." + model_name + " as net_symbol")

        if args.lr == -1:
            lr = config.LR_WM
        else:
            lr = args.lr
        mask_root = os.path.join(config.CACHE_PATH, config.FINAL_WEB_MASK_FOLDER)
        im_root = config.WEB_IMAGE_FOLDER
        flist_path = os.path.join(config.CACHE_PATH, config.WEB_IMAGE_FLIST)
        crop_size = config.CROP_SIZE_WM
        scale_range = config.SCALE_RANGE_WM
        label_shrink_scale = 1.0 / config.DOWN_SAMPLE_VALUE_WM
        epoch_size = config.EPOCH_SIZE_WM
        max_epoch = config.MAX_EPOCH_WM
        batch_size = config.BATCH_SIZE_WM
        lr_decay = config.LR_DECAY_WM
        use_g_labels=False

    else:
        model_name = "final_fcn_%s" % config.BASE_NET
        exec ("import cores.symbols." + model_name + " as net_symbol")
        mask_root = os.path.join(config.CACHE_PATH, config.FINAL_VOC_MASK_FOLDER)
        im_root = os.path.join(config.DATASET_PATH, config.VOC_TRAIN_IM_FOLDER)
        if args.lr == -1:
            lr = config.LR_FM
        else:
            lr = args.lr
        flist_path = os.path.join(config.DATASET_PATH, config.VOC_TRAIN_LIST)
        crop_size = config.CROP_SIZE_FM
        scale_range = config.SCALE_RANGE_FM
        label_shrink_scale = 1.0 / config.DOWN_SAMPLE_VALUE_FM
        epoch_size = config.EPOCH_SIZE_FM
        max_epoch = config.MAX_EPOCH_FM
        batch_size = config.BATCH_SIZE_FM
        lr_decay = config.LR_DECAY_FM
        use_g_labels=True


    model_prefix = os.path.join(config.SNAPSHOT_FOLDER, model_name)
    epoch = args.epoch
    mem_mirror = config.MEM_MIRROR
    class_num = config.CLASS_NUM
    workspace = config.WORKSPACE
    init_weight_file = "models/%s.params" % config.BASE_NET
    rgb_mean = config.MEAN_RGB
    wd = config.WD
    momentum = config.MOMENTUM

    param_list = ["ctx", "epoch", "lr", "mem_mirror", "model_prefix", "class_num", "workspace", "init_weight_file",
                  "im_root", "mask_root", "flist_path", "use_g_labels", "rgb_mean", "crop_size", "scale_range",
                  "label_shrink_scale", "epoch_size", "max_epoch", "batch_size", "wd",
                  "momentum", "lr_decay", "model_name"]
    result_str = "parameters\t"
    for param in param_list:
        result_str += "%s: %s\t" % (param, eval(param))
    logging.info(result_str)
    logging.info("start training the %s model" % args.model)

    train_seg_wrapper(ctx=ctx, epoch=epoch, lr=lr, mem_mirror=mem_mirror,
                      model_prefix=model_prefix, symbol=net_symbol, class_num=class_num,
                      workspace=workspace, init_weight_file=init_weight_file,
                      im_root=im_root, mask_root=mask_root, flist_path=flist_path,
                      use_g_labels=use_g_labels, rgb_mean=rgb_mean, crop_size=crop_size,
                      scale_range=scale_range, label_shrink_scale=label_shrink_scale,
                      epoch_size=epoch_size, max_epoch=max_epoch,
                      batch_size=batch_size,
                      wd=wd, momentum=momentum, lr_decay=lr_decay)

