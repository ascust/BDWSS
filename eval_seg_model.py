import argparse
from cores.seg_wrapper import test_seg_wrapper
import os
import mxnet as mx
import logging
import cores.config as config
import cores.utils.misc as misc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpu", default="0",
                        help="Device index.")
    parser.add_argument("--epoch", default=0, type=int,
                        help="snapshot name for evaluation")
    parser.add_argument("--savemask", help="whether save the prediction masks.",
                        action="store_true")
    parser.add_argument("--savescoremap", help="whether save the prediction scoremaps.",
                        action="store_true")
    parser.add_argument("--model", default="web",
                        help="evaluate the init model or final model. either \"web\" or \"final\"")
    parser.add_argument("--crf", help="whether use crf for post processing.",
                        action="store_true")
    args = parser.parse_args()

    ctx = [mx.gpu(int(args.gpu))]
    misc.my_mkdir(config.OUTPUT_FOLDER)
    misc.my_mkdir(config.LOG_FOLDER)

    log_file_name = os.path.join(config.LOG_FOLDER, "eval_model.log")
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    assert args.model == "web" or args.model == "final"

    if args.model == "web":
        model_name = "web_fcn_%s" % config.BASE_NET
        exec("import cores.symbols."+ model_name +" as net_symbol")
    else:
        model_name = "final_fcn_%s" % config.BASE_NET
        exec("import cores.symbols."+ model_name +" as net_symbol")


    epoch = args.epoch
    output_folder = config.OUTPUT_FOLDER
    save_mask = args.savemask
    save_scoremap = args.savescoremap
    class_num = config.CLASS_NUM
    workspace = config.WORKSPACE
    snapshot_folder = config.SNAPSHOT_FOLDER
    max_dim = config.MAX_INPUT_DIM
    im_root = os.path.join(config.DATASET_PATH, config.VOC_VAL_IM_FOLDER)
    mask_root = os.path.join(config.DATASET_PATH, config.VOC_VAL_MASK_FOLDER)
    flist_path = os.path.join(config.DATASET_PATH, config.VOC_VAL_LIST)
    rgb_mean = config.MEAN_RGB
    scale_list = config.EVAL_SCALE_LIST
    interp = config.INTERP_METHOD
    use_crf = args.crf
    crf_params = {}
    crf_params["pos_xy_std"] = config.CRF_POS_XY_STD
    crf_params["pos_w"] = config.CRF_POS_W
    crf_params["bi_xy_std"] = config.CRF_BI_XY_STD
    crf_params["bi_rgb_std"] = config.CRF_BI_RGB_STD
    crf_params["bi_w"] = config.CRF_BI_W

    param_list = ["epoch", "ctx", "output_folder", "model_name", "save_mask", "save_scoremap",
                  "class_num", "workspace", "snapshot_folder", "max_dim", "im_root", "mask_root",
                  "flist_path", "rgb_mean", "scale_list", "use_crf", "crf_params"]
    result_str = "parameters\t"
    for param in param_list:
        result_str += "%s: %s\t" % (param, eval(param))
    logging.info(result_str)

    test_seg_wrapper(epoch=epoch, ctx=ctx, output_folder=output_folder, model_name=model_name,
                     save_mask=save_mask, save_scoremap=save_scoremap, net_symbol=net_symbol,
                     class_num=class_num, workspace=workspace,
                     snapshot_folder=snapshot_folder, max_dim=max_dim,
                     im_root=im_root, mask_root=mask_root, flist_path=flist_path,
                     rgb_mean=rgb_mean, scale_list=scale_list, class_names=misc.get_voc_class_names(),
                     use_crf=use_crf, crf_params=crf_params)




