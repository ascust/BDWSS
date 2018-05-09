import argparse
from cores.sec_wrapper import create_SEC_cue, train_sec_wrapper
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
    parser.add_argument("--model", default="init",
                        help="Train for Init-SEC or Web-SEC. either \"init\" or \"web\"")

    args = parser.parse_args()
    misc.my_mkdir(config.SNAPSHOT_FOLDER)
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = str(config.CPU_WORKER_NUM)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    assert args.model in ["init", "web"], "wrong model type. Should be either \"init\" or \"web\""
    log_file_name = os.path.join(config.LOG_FOLDER, "train_SEC_%s_model.log"%args.model)
    if os.path.exists(log_file_name) and args.epoch==0:
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)


    if args.model == "init":
        bg_cue_file = os.path.join(config.CACHE_PATH, config.BG_CUE_FILE_INISEC)
        fg_cue_file = os.path.join(config.CACHE_PATH, config.FG_CUE_FILE_INITSEC)
        multi_lable_file = os.path.join(config.DATASET_PATH, config.VOC_TRAIN_MULTI_FILE)
        output_cue_file = os.path.join(config.CACHE_PATH, config.CUE_FILE_INITSEC)
        logging.info("generating cue file")
        create_SEC_cue(bg_cue_file=bg_cue_file, fg_cue_file=fg_cue_file,
                       multi_lable_file=multi_lable_file, output_cue_file=output_cue_file)
        logging.info("cue file generated")
        lr = config.LR_INITSEC
        im_root = os.path.join(config.DATASET_PATH, config.VOC_TRAIN_IM_FOLDER)
        epoch_size = config.EPOCH_SIZE_INITSEC
        batch_size = config.BATCH_SIZE_INITSEC
        lr_decay = config.LR_DECAY_INITSEC
        SC_only = False
    else:
        bg_cue_file = os.path.join(config.CACHE_PATH, config.BG_CUE_FILE_WEBSEC)
        fg_cue_file = os.path.join(config.CACHE_PATH, config.FG_CUE_FILE_WEBSEC)
        multi_lable_file = os.path.join(config.CACHE_PATH, config.WEB_IMAGE_LABEL_FILE)
        output_cue_file = os.path.join(config.CACHE_PATH, config.CUE_FILE_WEBSEC)
        logging.info("generating cue file")
        create_SEC_cue(bg_cue_file=bg_cue_file, fg_cue_file=fg_cue_file,
                       multi_lable_file=multi_lable_file, output_cue_file=output_cue_file)
        logging.info("cue file generated")
        lr = config.LR_WEBSEC
        im_root = config.WEB_IMAGE_FOLDER
        epoch_size = config.EPOCH_SIZE_WEBSEC
        batch_size = config.BATCH_SIZE_WEBSEC
        lr_decay = config.LR_DECAY_WEBSEC
        if config.BASE_NET == "vgg16":
            SC_only = True
        elif config.BASE_NET == "resnet50":
            SC_only = False

    model_name = "SEC_%s" % config.BASE_NET
    exec ("import cores.symbols." + model_name + " as net_symbol")
    model_prefix = os.path.join(config.SNAPSHOT_FOLDER, "%s_%s"%(model_name, args.model))
    epoch = args.epoch
    mem_mirror = config.MEM_MIRROR
    class_num = config.CLASS_NUM
    workspace = config.WORKSPACE
    input_size = config.INPUT_SIZE_SEC
    init_weight_file = "models/%s.params" % config.BASE_NET
    rgb_mean = config.MEAN_RGB
    down_sample = config.DOWN_SAMPLE_SEC
    wd = config.WD
    momentum = config.MOMENTUM
    q_fg = config.Q_FG
    q_bg = config.Q_BG


    param_list = ["ctx", "epoch", "lr", "mem_mirror", "model_prefix", "output_cue_file", "im_root",
                  "epoch_size", "batch_size", "lr_decay", "class_num", "workspace",
                  "input_size", "init_weight_file", "rgb_mean", "down_sample",
                  "wd", "momentum"]
    result_str = "parameters\t"
    for param in param_list:
        result_str += "%s: %s\t" % (param, eval(param))
    logging.info(result_str)
    logging.info("start training model %s_%s" % (model_name, args.model))
    train_sec_wrapper(ctx=ctx, epoch=epoch, lr=lr, mem_mirror=mem_mirror, model_prefix=model_prefix,
                     symbol=net_symbol, class_num=class_num, workspace=workspace, init_weight_file=init_weight_file,
                     im_root=im_root, rgb_mean=rgb_mean, im_size=input_size, label_shrink_scale=1.0/down_sample,
                     cue_file_path=output_cue_file, epoch_size=epoch_size, max_epoch=1, batch_size=batch_size,
                     wd=wd, momentum=momentum, lr_decay=lr_decay, q_fg=q_fg,
                     q_bg=q_bg, SC_only=SC_only)

