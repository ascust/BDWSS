#train multi-label classification for foreground cues for SEC models.
import argparse
import mxnet as mx
import cores.utils.misc as misc
import os
import cores.config as config
import logging
from cores.train_multi_wrapper import train_multi_wrapper
from cores.generate_fg_cues import generate_fg_cues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpus", default="0",
                        help="Device indices.")
    parser.add_argument("--model", default="init",
                        help="Train for Init-SEC or Web-SEC. either \"init\" or \"web\"")

    args = parser.parse_args()
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = str(config.CPU_WORKER_NUM)
    misc.my_mkdir(config.LOG_FOLDER)
    misc.my_mkdir(config.SNAPSHOT_FOLDER)
    misc.my_mkdir(config.CACHE_PATH)
    assert args.model in ["init", "web"], "wrong model type. Should be either \"init\" or \"web\""
    log_file_name = os.path.join(config.LOG_FOLDER, "train_fg_cue_net_%s.log"%args.model)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    if args.model == "init":
        im_folder = os.path.join(config.DATASET_PATH, config.VOC_TRAIN_IM_FOLDER)
        multi_label_file = os.path.join(config.DATASET_PATH, config.VOC_TRAIN_MULTI_FILE)
        epoch_size = config.EPOCH_SIZE_FG_INITSEC
        batch_size = config.BATCH_SIZE_FG_INITSEC
        lr = config.LR_FG_INITSEC
        lr_decay = config.LR_DECAY_FG_INITSEC
        cue_file = os.path.join(config.CACHE_PATH, config.FG_CUE_FILE_INITSEC)
    else:
        im_folder = config.WEB_IMAGE_FOLDER
        multi_label_file = os.path.join(config.CACHE_PATH, config.WEB_IMAGE_LABEL_FILE)
        epoch_size = config.EPOCH_SIZE_FG_WEBSEC
        batch_size = config.BATCH_SIZE_FG_WEBSEC
        lr = config.LR_FG_WEBSEC
        lr_decay = config.LR_DECAY_FG_WEBSEC
        cue_file = os.path.join(config.CACHE_PATH, config.FG_CUE_FILE_WEBSEC)

    mem_mirror = config.MEM_MIRROR
    if config.MEM_MIRROR:
        os.environ["MXNET_BACKWARD_DO_MIRROR"] = "1"
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    snapshot_prefix = os.path.join(config.SNAPSHOT_FOLDER, "fg_cue_net_%s"%args.model)
    model_file = os.path.join(config.SNAPSHOT_FOLDER, "fg_cue_net_%s-1.params" % args.model)
    init_weight_file = "models/%s.params" % config.BASE_NET
    class_num = config.CLASS_NUM - 1 #exclude bg class
    rgb_mean = config.MEAN_RGB
    input_size = config.INPUT_SIZE_SEC
    wd = config.WD
    momentum = config.MOMENTUM
    workspace = config.WORKSPACE
    saliency_th = config.SALIENCY_TH_FG
    model_name = "fg_cue_%s" % config.BASE_NET
    exec ("import cores.symbols." + model_name + " as net_symbol")
    output_size = config.INPUT_SIZE_SEC/config.DOWN_SAMPLE_SEC
    #check shape
    _, outshape, _ = net_symbol.create_body().infer_shape(data=(1, 3, config.INPUT_SIZE_SEC, config.INPUT_SIZE_SEC))
    assert outshape[0][2] == output_size, "output shapes do not match."

    param_list = ["ctx", "mem_mirror", "im_folder", "multi_label_file", "epoch_size", "batch_size", "lr", "lr_decay", "cue_file",
                  "snapshot_prefix", "model_file", "init_weight_file", "class_num", "rgb_mean", "input_size",
                    "wd", "momentum", "workspace", "saliency_th", "model_name",
                  "output_size"]
    result_str = "parameters\t"
    for param in param_list:
        result_str += "%s: %s\t" % (param, eval(param))
    logging.info(result_str)

    logging.info("start training fg cues for %s SEC."%args.model)

    train_multi_wrapper(ctx=ctx, symbol=net_symbol, snapshot_prefix=snapshot_prefix,
                        init_weight_file=init_weight_file, im_folder=im_folder,
                        multi_label_file=multi_label_file,
                        class_num=class_num, rgb_mean=rgb_mean,
                        epoch_size=epoch_size, max_epoch=1, input_size=input_size,
                        batch_size=batch_size,
                        lr=lr, wd=wd, momentum=momentum, lr_decay=lr_decay, workspace=workspace)

    logging.info("start generating fg cue file for %s SEC."%args.model)

    generate_fg_cues(ctx=ctx, image_root=im_folder, multilabel_file=multi_label_file,
                     rgb_mean=rgb_mean, symbol=net_symbol, class_num=class_num,
                     model_file=model_file, input_size=input_size, batch_size=batch_size,
                     output_size=output_size, saliency_th=saliency_th, workspace=workspace, cue_file_path=cue_file)




