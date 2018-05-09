import cores.config as config
import os
import cores.utils.misc as misc
import argparse
import shutil
from cores.utils.voc_cmap import get_cmap
from PIL import Image
import numpy as np

def generate_refined_masks(orig_mask_folder, tmp_gc_results_folder, output_folder, th, use_ignore, flist_path):
	cmap = get_cmap()

	count = 0
	total = len(os.listdir(tmp_gc_results_folder))
	flist = []
	for i in os.listdir(tmp_gc_results_folder):
		if i.endswith(".png") and not i.startswith("."):
			flist.append(i.replace(".png", ""))
			l = np.array(Image.open(os.path.join(tmp_gc_results_folder, i)))
			l = l>=(th*l.max())
			orig_l = np.array(Image.open(os.path.join(orig_mask_folder, i)), dtype=np.uint8)
			assert l.shape == orig_l.shape
			if use_ignore:
				orig_l[(orig_l>0) & (l==0)] = 255
			else:
				orig_l[l==0] = 0
			out_img = Image.fromarray(orig_l)
			out_img.putpalette(cmap)
			out_img.save(os.path.join(output_folder, i))
			count += 1
			print "processed %s\t%d/%d" % (i, count, total)
	with open(flist_path, "w") as f:
		for i in flist:
			f.write("%s\n"%i)
	print "done!"


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Training parameters")
	parser.add_argument("--nogc", help="Do not include Grabcut refinement results.",
                        action="store_true")
	args = parser.parse_args()

	mask_folder = os.path.join(config.CACHE_PATH, config.WEB_MASK_FOLDER_WEBSEC)
	final_output_folder = os.path.join(config.CACHE_PATH, config.FINAL_WEB_MASK_FOLDER)
	misc.my_mkdir(final_output_folder)
	flist_path = os.path.join(config.CACHE_PATH, config.WEB_IMAGE_FLIST)

	if args.nogc:
		#no refinement. Simply copy the files and create a list.
		count = 0
		total = len(os.listdir(mask_folder))
		flist = []
		for i in os.listdir(mask_folder):
			if i.endswith(".png") and not i.startswith("."):
				flist.append(i.replace(".png", ""))
				shutil.copyfile(os.path.join(mask_folder, i), os.path.join(final_output_folder, i))
				count += 1
				print "processed %s\t%d/%d" % (i, count, total)
		with open(flist_path, "w") as f:
		    for i in flist:
				f.write("%s\n"%i)
		print "done!"

	else:
		tmp_gc_results = os.path.join(config.CACHE_PATH, config.TMP_GC_RESULTS_FOLDER)
		generate_refined_masks(orig_mask_folder=mask_folder, tmp_gc_results_folder=tmp_gc_results,
	               output_folder=final_output_folder, th=config.FG_TH_GC,
	               use_ignore=config.USE_IGNORE_GC, flist_path=flist_path)
    
    
    
    
    

    