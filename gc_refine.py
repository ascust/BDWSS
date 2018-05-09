import cPickle as pickle
from PIL import Image
import cv2
import random
import os
import numpy as np
import cores.config as config
import cores.utils.misc as misc


def gc_refine(im_folder, mask_folder, output_folder, label_file, max_sample, max_trial, offset, min_dim_th, margin):
    label_dict = pickle.load(open(label_file, 'rb'))

    flist = label_dict.keys()
    random.shuffle(flist)

    for f in flist:
        if os.path.exists(os.path.join(output_folder, f + ".png")) or f.startswith("0"):
            print "skipped %s" % f
            continue
        print "processing %s" % f
        class_index = label_dict[f][0] + 1
        im_name = f + ".jpg"
        l_name = f + ".png"
        im = np.array(Image.open(os.path.join(im_folder, im_name)))
        l = np.array(Image.open(os.path.join(mask_folder, l_name)))
        h, w = l.shape

        init_mask = np.zeros(l.shape, dtype=np.uint8)
        init_mask[l > 0] = 3
        init_mask[l == 0] = 2
        tmp = (l == class_index)
        if tmp.sum() < int(w * h * min_dim_th * min_dim_th):
            print "too few pixels, skipped %s" % f
            continue

        sum0 = tmp.sum(0)
        left = np.where(sum0 > 0)[0][0]
        right = np.where(sum0 > 0)[0][-1]
        sum1 = tmp.sum(1)
        top = np.where(sum1 > 0)[0][0]
        bottom = np.where(sum1 > 0)[0][-1]
        rect = (left, top, right - left, bottom - top)
        rect_w = rect[2]
        rect_h = rect[3]
        left = rect[0]
        top = rect[1]
        right = rect[0] + rect_w
        bottom = rect[1] + rect_h

        final_mask = np.zeros(l.shape, dtype=np.uint8)

        count = 0
        val_count = 0
        while True:
            count += 1
            if count >= max_trial:
                print "reaches max trials, break."
                break
            print "trial: %d" % (count)

            left_offset = int(rect_w * (2 * offset * random.random() - offset))
            top_offset = int(rect_h * (2 * offset * random.random() - offset))
            right_offset = int(rect_w * (2 * offset * random.random() - offset))
            bottom_offset = int(rect_h * (2 * offset * random.random() - offset))
            # print left_offset, top_offset, right_offset, bottom_offset

            new_left = max(margin, left + left_offset)
            new_top = max(margin, top + top_offset)
            new_right = min(w - margin, right + right_offset)
            new_bottom = min(h - margin, bottom + bottom_offset)
            new_rect = (new_left, new_top, new_right - new_left, new_bottom - new_top)

            if new_rect[2] < w * min_dim_th or new_rect[3] < h * min_dim_th:
                print "bad rect, skipped"
                continue
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            mask = init_mask.copy()
            mask[0:new_top, :] = 0
            mask[new_bottom:, :] = 0
            mask[:, 0:new_left] = 0
            mask[:, new_right:] = 0

            cv2.grabCut(im[:, :, ::-1], mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            mask[mask == 2] = 0
            mask[mask > 0] = 1

            final_mask += mask
            val_count += 1
            if val_count >= max_sample:
                print "enough samples, break."
                break
        if val_count < max_sample:
            print "bad result"
        else:
            output_im = Image.fromarray(final_mask)
            output_im.save(os.path.join(output_folder, l_name))
            print "processed %s" % f


if __name__ == "__main__":
    mask_folder = os.path.join(config.CACHE_PATH, config.WEB_MASK_FOLDER_WEBSEC)
    tmp_output_folder = os.path.join(config.CACHE_PATH, config.TMP_GC_RESULTS_FOLDER)
    label_file = os.path.join(config.CACHE_PATH, config.WEB_IMAGE_LABEL_FILE)
    misc.my_mkdir(tmp_output_folder)
    #refine masks using grabcut
    gc_refine(im_folder=config.WEB_IMAGE_FOLDER, mask_folder=mask_folder,output_folder=tmp_output_folder,
              label_file=label_file, max_sample=config.MAX_SAMPLE_GC, max_trial=config.MAX_TRIAL_GC,
              offset=config.OFFSET_GC, min_dim_th=config.MIN_DIM_TH_GC, margin=config.MARGIN_GC)
