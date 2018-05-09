import argparse
import os
import time
import logging
import cores.config as config
import cores.utils.misc as misc
import subprocess

def get_untested_list(snapshot_folder, output_folder, model_name):
    untested_list = []
    snapshot_files = os.listdir(config.SNAPSHOT_FOLDER)
    for f in snapshot_files:
        if f.endswith(".params") and model_name in f:
            epoch_num = int(f[f.rfind('-')+1:f.rfind('.params')])
            output_folder = model_name + "_epoch" + str(epoch_num)
            if os.path.exists(os.path.join(config.OUTPUT_FOLDER, output_folder)):
                continue
            untested_list.append(epoch_num)
    return untested_list

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
                        help="evaluate the web model or final model. either \"web\" or \"final\"")
    parser.add_argument("--crf", help="whether use crf for post processing.",
                        action="store_true")
    args = parser.parse_args()


    misc.my_mkdir(config.OUTPUT_FOLDER)

    log_file_name = os.path.join(config.LOG_FOLDER, "eval_model.log")
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    assert args.model == "web" or args.model == "final"
    if args.model == "web":
        model_name = "web_fcn_%s" % config.BASE_NET
    elif args.model == "final":
        model_name = "final_fcn_%s" % config.BASE_NET

    if args.epoch == 0:
        while True:
            untested_list = get_untested_list(config.SNAPSHOT_FOLDER, config.OUTPUT_FOLDER, model_name=model_name)
            for epoch_num in untested_list:
                cmd_str = "python eval_seg_model.py --epoch %d --gpu %s --model %s" % (epoch_num, args.gpu, args.model)
                if args.savemask:
                    cmd_str += " --savemask"
                if args.savescoremap:
                    cmd_str += " --savescoremap"
                if args.crf:
                    cmd_str += " --crf"
                subprocess.call(cmd_str, shell=True)
            untested_list = get_untested_list(config.SNAPSHOT_FOLDER, config.OUTPUT_FOLDER, model_name=model_name)
            if len(untested_list)>0:
                continue
            else:
                print "Waiting for %f hours" % config.EVAL_WAIT_TIME
                time.sleep(config.EVAL_WAIT_TIME*3600)
                untested_list = get_untested_list(config.SNAPSHOT_FOLDER, config.OUTPUT_FOLDER, model_name=model_name)
                if len(untested_list)>0:
                    continue
                else:
                    break
        print "Done! No more testing."
    else:
        cmd_str = "python eval_seg_model.py --epoch %d --gpu %s --model %s" % (args.epoch, args.gpu, args.model)
        if args.savemask:
            cmd_str += " --savemask"
        if args.savescoremap:
            cmd_str += " --savescoremap"
        if args.crf:
            cmd_str += " --crf"
        subprocess.call(cmd_str, shell=True)




