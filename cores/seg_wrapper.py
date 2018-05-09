import logging
import os
import utils.misc as misc
import mxnet as mx
from data.SegTrainingIter import SegTrainingIter
import utils.callbacks as callbacks
import utils.metrics as metrics
from utils.voc_cmap import get_cmap
from data.InferenceDataProducer import InferenceDataProducer
import numpy as np
from PIL import Image
from cores.utils.CRF import CRF

def train_seg_wrapper(ctx, epoch, lr, mem_mirror, model_prefix, symbol, class_num, workspace, init_weight_file,
                      im_root, mask_root, flist_path, use_g_labels, rgb_mean, crop_size, scale_range, label_shrink_scale,
                      epoch_size, max_epoch, batch_size, wd, momentum, lr_decay):

    if mem_mirror:
        os.environ["MXNET_BACKWARD_DO_MIRROR"] = "1"
    arg_dict = {}
    aux_dict = {}
    if use_g_labels:
        seg_net = symbol.create_training(class_num=class_num, gweight=1.0/batch_size, workspace=workspace)
    else:
        seg_net = symbol.create_training(class_num=class_num, workspace=workspace)
    if epoch == 0:
        if not os.path.exists(init_weight_file):
            logging.warn("No model file found at %s. Start from scratch!" % init_weight_file)
        else:
            arg_dict, aux_dict, _ = misc.load_checkpoint(init_weight_file)
            param_types = ["_weight", "_bias", "_gamma", "_beta", "_moving_mean", "_moving_var"]
            #copy params for global branch
            if use_g_labels:
                for arg in arg_dict.keys():
                    for param_type in param_types:
                        if param_type in arg:
                            arg_name = arg[:arg.rfind(param_type)]
                            arg_dict[arg_name + "_g" + param_type] = arg_dict[arg].copy()
                            if arg_name in ["fc6", "fc7"]:
                                arg_dict[arg_name + "_1" + param_type] = arg_dict[arg].copy()
                                arg_dict[arg_name + "_2" + param_type] = arg_dict[arg].copy()
                                arg_dict[arg_name + "_3" + param_type] = arg_dict[arg].copy()
                                arg_dict[arg_name + "_4" + param_type] = arg_dict[arg].copy()
                            break
                for aux in aux_dict.keys():
                    for param_type in param_types:
                        if param_type in aux:
                            aux_name = aux[:aux.rfind(param_type)]
                            aux_dict[aux_name + "_g" + param_type] = aux_dict[aux].copy()
                            break
    else:
        arg_dict, aux_dict, _ = misc.load_checkpoint(model_prefix, epoch)

    data_iter = SegTrainingIter(
        im_root=im_root,
        mask_root=mask_root,
        file_list_path=flist_path,
        provide_g_labels=use_g_labels,
        class_num=class_num,
        rgb_mean=rgb_mean,
        crop_size=crop_size,
        shuffle=True,
        scale_range=scale_range,
        label_shrink_scale=label_shrink_scale,
        random_flip=True,
        data_queue_size=8,
        epoch_size=epoch_size,
        batch_size=batch_size,
        round_batch=True
    )


    initializer = mx.initializer.Normal()
    initializer.set_verbosity(True)

    if use_g_labels:
        mod = mx.mod.Module(seg_net, context=ctx, label_names=["softmax_label", "g_logistic_label"])
    else:
        mod = mx.mod.Module(seg_net, context=ctx, label_names=["softmax_label"])
    mod.bind(data_shapes=data_iter.provide_data,
            label_shapes=data_iter.provide_label)
    mod.init_params(initializer=initializer, arg_params=arg_dict, aux_params=aux_dict, allow_missing=(epoch == 0))

    opt_params = {"learning_rate":lr,
                "wd": wd,
                'momentum': momentum,
                'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=lr_decay, factor=0.1),
                'rescale_grad': 1.0/len(ctx)}

    if use_g_labels:
        eval_metrics = [metrics.Accuracy(), metrics.Loss(), metrics.MultiLogisticLoss(l_index=1, p_index=1)]
    else:
        eval_metrics = [metrics.Accuracy(), metrics.Loss()]
    mod.fit(data_iter,
            optimizer="sgd",
            optimizer_params=opt_params,
            num_epoch=max_epoch,
            epoch_end_callback=callbacks.module_checkpoint(model_prefix),
            batch_end_callback=callbacks.Speedometer(batch_size, frequent=10),
            eval_metric=eval_metrics,
            begin_epoch=epoch+1)

def test_seg_wrapper(epoch, ctx, output_folder, model_name, save_mask, save_scoremap, net_symbol, class_num, workspace,
                     snapshot_folder, max_dim, im_root, mask_root, flist_path, rgb_mean, scale_list,
                     class_names, use_crf=False, crf_params=None):

    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"]="0"

    crf_obj = None
    if use_crf:
        assert crf_params is not None
        crf_obj = CRF(pos_xy_std=crf_params["pos_xy_std"], pos_w=crf_params["pos_w"], bi_xy_std=crf_params["bi_xy_std"],
                      bi_rgb_std=crf_params["bi_rgb_std"], bi_w=crf_params["bi_w"], scale_factor=1.0)


    epoch_str = str(epoch)
    misc.my_mkdir(output_folder)
    misc.my_mkdir(os.path.join(output_folder, model_name + "_epoch" + epoch_str))

    if save_mask:
        misc.my_mkdir(os.path.join(output_folder, model_name + "_epoch" + epoch_str, "masks"))
    if save_scoremap:
        misc.my_mkdir(os.path.join(output_folder, model_name + "_epoch" + epoch_str, "scoremaps"))

    cmap = get_cmap()

    model_prefix = os.path.join(snapshot_folder, model_name)
    seg_net = net_symbol.create_infer(class_num, workspace)
    arg_dict, aux_dict, _ = misc.load_checkpoint(model_prefix, epoch)


    mod = mx.mod.Module(seg_net, label_names=[], context=ctx)
    mod.bind(data_shapes=[("data", (1, 3, max_dim, max_dim))],
             for_training=False, grad_req="null")
    initializer = mx.init.Normal()
    initializer.set_verbosity(True)
    mod.init_params(initializer=initializer, arg_params=arg_dict, aux_params=aux_dict, allow_missing=True)

    data_producer = InferenceDataProducer(
            im_root=im_root,
            mask_root=mask_root,
            flist_path=flist_path,
            rgb_mean=rgb_mean,
            scale_list=scale_list)

    nbatch = 0
    eval_metrics = [metrics.IOU(class_num, class_names)]
    logging.info("In evaluation...")

    while True:
        data = data_producer.get_data()
        if data is None:
            break
        im_list = data[0]

        label = data[1].squeeze()
        file_name = data[2]
        ori_im = data[3]
        final_scoremaps = mx.nd.zeros((class_num, label.shape[0], label.shape[1]))

        for im in im_list:
            mod.reshape(data_shapes=[("data", im.shape)])
            mod.forward(mx.io.DataBatch(data=[mx.nd.array(im)]))

            score = mx.nd.transpose(mod.get_outputs()[0].copyto(mx.cpu()), [0, 2, 3, 1])
            score = mx.nd.reshape(score, (score.shape[1], score.shape[2], score.shape[3]))
            up_score = mx.nd.transpose(mx.image.imresize(score, label.shape[1], label.shape[0], interp=1), [2, 0, 1])

            final_scoremaps += up_score

        final_scoremaps = mx.nd.log(final_scoremaps)
        final_scoremaps = final_scoremaps.asnumpy()
        if use_crf:
            assert crf_params is not None
            final_scoremaps = crf_obj.inference(ori_im.asnumpy(), final_scoremaps)

        pred_label = final_scoremaps.argmax(0)

        for eval in eval_metrics:
            eval.update(label, pred_label)

        if save_mask:
            out_img = np.uint8(pred_label)
            out_img = Image.fromarray(out_img)
            out_img.putpalette(cmap)
            out_img.save(os.path.join(output_folder, model_name + "_epoch" + epoch_str, "masks", file_name+".png"))
        if save_scoremap:
            np.save(os.path.join(output_folder, model_name + "_epoch" + epoch_str, "scoremaps", file_name), final_scoremaps)

        nbatch += 1
        if nbatch % 10 == 0:
            print "processed %dth batch" % nbatch

    logging.info("Epoch [%d]: " % epoch)
    for m in eval_metrics:
        logging.info("[overall] [%s: %.4f]" % (m.get()[0], m.get()[1]))
        if m.get_class_values() is not None:
            scores = "[perclass] ["
            for v in m.get_class_values():
                scores += "%s: %.4f\t" % (v[0], v[1])
            scores += "]"
            logging.info(scores)
