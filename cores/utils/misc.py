import mxnet as mx
import cv2
import numpy as np
import os
import logging

#resize feature map that has (n, h, w) to (n, targetH, targetW)
#interp 1 for bilinear and 2 for bicubic
def resize_feat(feat, targetH, targetW, interp=1):
    assert len(feat.shape) == 3
    output = np.zeros((feat.shape[0], targetH, targetW), dtype=np.float32)
    for i in range(feat.shape[0]):
        output[i][:] = cv2.resize(feat[i], (targetW, targetH), interpolation=interp)
    return output

def my_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

#load checkpoint
def load_checkpoint(prefix, epoch=None, load_symbol=False):
    symbol = None
    if load_symbol:
        symbol = mx.sym.load('%s-symbol.json' % prefix)
    if epoch is None:
        save_dict = mx.nd.load(prefix)
    else:
        save_dict = mx.nd.load('%s-%d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (arg_params, aux_params, symbol)

#sava checkpoint
def save_checkpoint(prefix, epoch, symbol, arg_params, aux_params):
    if symbol is not None:
        symbol.save('%s-symbol.json' % prefix)

    save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    param_name = '%s-%d.params' % (prefix, epoch)
    mx.nd.save(param_name, save_dict)
    logging.info('Saved checkpoint to \"%s\"', param_name)

def get_voc_class_names():
    return [    'bk', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

def pad_image(img_array, downsample_scale):
    r = img_array.shape[-2]
    c = img_array.shape[-1]
    orig_size = (r, c)
    scale = int(downsample_scale)
    r_pad = 0
    c_pad = 0
    if r % scale > 0:
        r_pad = (r/scale+1)*scale - r
    if c % scale > 0:
        c_pad = (c/scale+1)*scale - c
    if r_pad>0 or c_pad>0:
        img_array = np.lib.pad(img_array, ((0,0),(0,0),(0,r_pad), (0, c_pad)), 'constant', constant_values=0)
    return img_array, orig_size
