import os
import cPickle as pickle
from PIL import Image
import numpy as np
import time
import cores.config as config
import datetime

mask_folder = os.path.join(config.CACHE_PATH, config.WEB_MASK_FOLDER_INITSEC)

class_file_queue = []
for i in range(config.CLASS_NUM):
    class_file_queue.append([])
count = 0
processed_count = 0
total = len(os.listdir(mask_folder))
start_time = time.time()
for file_name in os.listdir(mask_folder):
    if file_name.endswith(".png") and not file_name.startswith("."):
        count += 1
        name_noext = file_name.replace(".png", "")
        class_index = int(name_noext[:name_noext.find("_")])
        if class_index == 0:
            print "0_xxxx.png is for background. Skipped."
            continue
        if len(class_file_queue[class_index]) >= config.MAX_PER_CLASS:
            print "already reached the maximum for class %d, skipped %s!" % (class_index, file_name)
            continue
        try:
            mask = np.array(Image.open(os.path.join(mask_folder, file_name)))
        except Exception as e:
            print "error: %s" % e
            continue
        h, w = mask.shape


        fg_area_cur = (mask==class_index).sum()
        fg_area_other = np.logical_and(mask!=class_index, mask!=0).sum()
        fg_ratio_cur = float(fg_area_cur) / (w * h)
        fg_ratio_other = float(fg_area_other) / (w * h)


        if fg_ratio_cur >= config.FG_TH_CURCLASS_LOW and fg_ratio_cur <= config.FG_TH_CURCLASS_HI \
                and fg_ratio_other <= config.FG_TH_OTHER_HI:
            class_file_queue[class_index].append(name_noext)
        else:
            print "fg_ratio_cur: %f" % fg_ratio_cur
            print "fg_ratio_other: %f" % fg_ratio_other

        elapsed_time = time.time() - start_time
        eta = int(elapsed_time / count * (total - count))
        print "processed %s\t%d/%d\teta: %s" % (file_name, count, total, str(datetime.timedelta(seconds=eta)))
print "creating label file..."
data_dict = {}
for i in class_file_queue:
    for j in i:
        class_index = int(j[:j.find("_")])
        l = class_index - 1
        data_dict[j] = [l]

file_path = os.path.join(config.CACHE_PATH, config.WEB_IMAGE_LABEL_FILE)
pickle.dump(data_dict, open(file_path, 'wb'))
print "done! got %d in total" % len(data_dict.keys())
