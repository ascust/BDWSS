import logging
import os
import utils.misc as misc
import mxnet as mx
from data.SECTrainingIter import SECTrainingIter
import utils.callbacks as callbacks
import utils.metrics as metrics
import numpy as np
import cPickle as pickle

def train_sec_wrapper(ctx, epoch, lr, mem_mirror, model_prefix, symbol, class_num, workspace, init_weight_file,
                      im_root, rgb_mean, im_size, label_shrink_scale, cue_file_path,
                      epoch_size, max_epoch, batch_size, wd, momentum, lr_decay, q_fg, q_bg, SC_only=False):

    if mem_mirror:
        os.environ["MXNET_BACKWARD_DO_MIRROR"] = "1"
    arg_dict = {}
    aux_dict = {}
    outputsize = int(im_size*label_shrink_scale)
    seg_net = symbol.create_training(class_num=class_num, outputsize=outputsize, workspace=workspace, SC_only=SC_only)
    if epoch == 0:
        if not os.path.exists(init_weight_file):
            logging.warn("No model file found at %s. Start from scratch!" % init_weight_file)
        else:
            arg_dict, aux_dict, _ = misc.load_checkpoint(init_weight_file)
    else:
        arg_dict, aux_dict, _ = misc.load_checkpoint(model_prefix, epoch)
    #init weights for expand loss
    if not SC_only:
        arg_dict["fg_w"] = mx.nd.array(np.array([q_fg ** i for i in range(outputsize * outputsize - 1, -1, -1)])[None, None, :])
        arg_dict["bg_w"] = mx.nd.array(np.array([q_bg ** i for i in range(outputsize * outputsize - 1, -1, -1)])[None, :])

    data_iter = SECTrainingIter(
        im_root=im_root,
        cue_file_path=cue_file_path,
        class_num=class_num,
        rgb_mean=rgb_mean,
        im_size=im_size,
        shuffle=True,
        label_shrink_scale=label_shrink_scale,
        random_flip=True,
        data_queue_size=8,
        epoch_size=epoch_size,
        batch_size=batch_size,
        round_batch=True,
        SC_only=SC_only
    )

    initializer = mx.initializer.Normal()
    initializer.set_verbosity(True)

    if SC_only:
        mod = mx.mod.Module(seg_net, context=ctx, data_names=["data", "small_ims"], label_names=["cues"])
    else:
        mod = mx.mod.Module(seg_net, context=ctx, data_names=["data", "small_ims"], label_names=["labels", "cues"])

    mod.bind(data_shapes=data_iter.provide_data,
            label_shapes=data_iter.provide_label)
    mod.init_params(initializer=initializer, arg_params=arg_dict, aux_params=aux_dict, allow_missing=(epoch == 0))

    opt_params = {"learning_rate":lr,
                "wd": wd,
                'momentum': momentum,
                'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=lr_decay, factor=0.1),
                'rescale_grad': 1.0/len(ctx)}

    if SC_only:
        eval_metrics = [metrics.SEC_seed_loss(), metrics.SEC_constrain_loss()]
    else:
        eval_metrics = [metrics.SEC_seed_loss(), metrics.SEC_constrain_loss(), metrics.SEC_expand_loss()]
    mod.fit(data_iter,
            optimizer="sgd",
            optimizer_params=opt_params,
            num_epoch=max_epoch+1,
            epoch_end_callback=callbacks.module_checkpoint(model_prefix),
            batch_end_callback=callbacks.Speedometer(batch_size, frequent=10),
            eval_metric=eval_metrics,
            begin_epoch=epoch+1)

def create_SEC_cue(bg_cue_file, fg_cue_file, multi_lable_file, output_cue_file):
    with open(multi_lable_file, 'rb') as f:
        multi_label_dict = pickle.load(f)
    with open(fg_cue_file, 'rb') as f:
        fg_dict = pickle.load(f)
    with open(bg_cue_file, 'rb') as f:
        bg_dict = pickle.load(f)
    new_cue_dict = {}
    for f in multi_label_dict.keys():
        cues = fg_dict[f]
        bg_cues = bg_dict[f]
        new_cues = np.zeros((3, len(bg_cues[0]) + len(cues[0])), dtype=np.uint8)
        new_cues[0] = np.concatenate([np.zeros(bg_cues[0].shape), cues[0] + 1])
        new_cues[1] = np.concatenate([bg_cues[0], cues[1]])
        new_cues[2] = np.concatenate([bg_cues[1], cues[2]])
        new_cue_dict[f + "_cues"] = new_cues
        new_cue_dict[f + "_labels"] = np.array(multi_label_dict[f], dtype=np.uint8) + 1
    with open(output_cue_file, "wb") as f:
        pickle.dump(new_cue_dict, f)

