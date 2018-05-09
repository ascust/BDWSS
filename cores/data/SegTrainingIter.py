from Queue import Queue
from threading import Thread
from mxnet.io import DataIter, DataBatch
import mxnet as mx
import os
from PIL import Image
import numpy as np
import preprocessing
import random

class SegTrainingIter(DataIter):
    def __init__(self,
                 im_root,
                 mask_root,
                 file_list_path,
                 provide_g_labels,
                 class_num,
                 rgb_mean=(128, 128, 128),
                 crop_size=320,
                 shuffle=True,
                 scale_range=(0.7, 1.3),
                 label_shrink_scale=1/8.0,
                 random_flip=True,
                 data_queue_size=8,
                 epoch_size=-1,
                 batch_size=1,
                 round_batch=True):
        super(SegTrainingIter, self).__init__()
        self.rgb_mean = mx.nd.array(rgb_mean, dtype=np.uint8, ctx=mx.cpu()).reshape((1, 1, 3))
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        assert len(scale_range) == 2
        assert scale_range[1] >= scale_range[0]
        self.provide_g_labels = provide_g_labels
        self.class_num = class_num
        self.scale_range = scale_range
        self.label_shrink_scale = label_shrink_scale
        self.flist = None
        self.im_root = im_root
        self.mask_root = mask_root
        self.round_batch = round_batch

        self._load_flist(file_list_path)
        self.data_num = self.get_data_num()
        self.iter_count = 0
        self.cursor = 0
        self.reset()
        self.random_flip = random_flip
        if epoch_size == -1:
            self.epoch_size = int(self.data_num/self.batch_size)
        else:
            self.epoch_size = epoch_size

        self.flist_item_queue = Queue(maxsize=1000)
        list_producer = Thread(target=self._produce_flist_item)
        list_producer.daemon = True
        list_producer.start()
        self.data_queue = Queue(maxsize=data_queue_size)

        producer = Thread(target=self._produce_data)
        producer.daemon = True
        producer.start()


    def _produce_flist_item(self):
        while True:
            if self.cursor + self.batch_size < self.data_num:
                sub_list = self.flist[self.cursor:self.cursor+self.batch_size]
                self.cursor += self.batch_size
            else:
                if self.round_batch:
                    sub_list = self.flist[self.cursor:self.data_num]
                    sub_list += self.flist[0:(self.batch_size - len(sub_list))]
                    self.cursor = 0
                    if self.shuffle:
                        np.random.shuffle(self.flist)
                else:
                    if self.shuffle:
                        np.random.shuffle(self.flist)
                    sub_list = self.flist[0:self.batch_size]
                    self.cursor = self.batch_size
            self.flist_item_queue.put(sub_list)

    def _produce_data(self):
        while True:
            images = mx.nd.zeros((self.batch_size, 3, self.crop_size, self.crop_size))
            mask_dim = int(self.crop_size * self.label_shrink_scale)
            masks =  mx.nd.zeros((self.batch_size, mask_dim * mask_dim))
            if self.provide_g_labels:
                g_labels =  mx.nd.zeros((self.batch_size, self.class_num, 1, 1))
            sub_list = self.flist_item_queue.get()


            batch_images = []
            for image_path in list(sub_list):
                buf = mx.nd.array(
                    np.frombuffer(open(os.path.join(self.im_root, image_path+".jpg"), 'rb').read(), dtype=np.uint8),
                    dtype=np.uint8, ctx=mx.cpu())
                batch_images.append(mx.image.imdecode(buf))
            batch_labels = []
            for ind, label_path in enumerate(sub_list):
                mask = Image.open(os.path.join(self.mask_root, label_path+".png"))
                mask_arr = np.array(mask, dtype=np.uint8)
                if self.provide_g_labels:
                    g_l_arr = np.zeros(self.class_num, dtype=np.uint8)
                    ul = np.unique(mask_arr)
                    ul = ul[ul != 255]
                    g_l_arr[ul] = 1
                    g_l_arr = g_l_arr.reshape(-1, 1, 1)
                    g_labels[ind][:] = g_l_arr
                batch_labels.append(mask_arr)

            for ind in range(len(batch_images)):
                im_arr = batch_images[ind]
                l_arr = batch_labels[ind]
                r_start, c_start, new_crop_size = preprocessing.calc_crop_params(im_arr, self.scale_range,
                                                                                 self.crop_size)
                if self.random_flip:
                    im_arr, l_arr = preprocessing.random_flip(im_arr, l_arr)
                im_arr, l_arr = preprocessing.pad_image(im_arr, l_arr, new_crop_size, self.rgb_mean,
                                                        ignored_label=255)
                im_arr = im_arr[r_start:r_start + new_crop_size, c_start:c_start + new_crop_size, :]
                l_arr = l_arr[r_start:r_start + new_crop_size, c_start:c_start + new_crop_size]
                batch_images[ind] = im_arr
                batch_labels[ind] = l_arr

            l_dim = int(self.crop_size * self.label_shrink_scale)
            batch_images = [mx.image.imresize(im, self.crop_size, self.crop_size, interp=1) for im in batch_images]
            for ind in range(len(batch_labels)):
                mask_arr = batch_labels[ind]
                mask_arr = Image.fromarray(mask_arr).resize((l_dim, l_dim), Image.NEAREST)
                mask_arr = np.array(mask_arr, dtype=np.uint8)
                batch_labels[ind] = mask_arr
            for i in range(len(sub_list)):
                images[i][:] = mx.nd.transpose(batch_images[i], (2, 0, 1))
                masks[i][:] = batch_labels[i].reshape(-1)

            images -= mx.nd.reshape(self.rgb_mean, (1, 3, 1, 1)).astype(np.float32)

            if self.provide_g_labels:
                self.data_queue.put(DataBatch(data=[images], label=[masks, g_labels], pad=None, index=None))
            else:
                self.data_queue.put(DataBatch(data=[images], label=[masks], pad=None, index=None))





    def get_data_num(self):
        return len(self.flist)

    def _load_flist(self,
                   flist_path):
        with open(flist_path) as f:
            lines = f.readlines()
            self.flist = [i.strip() for i in lines]
            self.data_num = len(self.flist)
            if self.shuffle:
                random.shuffle(self.flist)


    def reset(self):
        self.iter_count = 0

    def iter_next(self):
        return self.iter_count < self.epoch_size

    def next(self):
        if self.iter_next():
            self.iter_count += 1
            return self.data_queue.get()
        else:
            raise StopIteration

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [("data", (self.batch_size, 3, self.crop_size, self.crop_size))]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        label_dim = int(self.crop_size * self.label_shrink_scale)
        if self.provide_g_labels:
            return [("softmax_label", (self.batch_size, label_dim*label_dim)),
                    ("g_logistic_label", (self.batch_size, self.class_num, 1, 1))]
        else:
            return [("softmax_label", (self.batch_size, label_dim*label_dim))]
