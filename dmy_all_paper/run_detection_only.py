# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: run.py
import os, cv2, time, sys
from pathlib import Path
from FCOS.configs.configs import fcos_cfgs
from FCOS.fcos_core.predictor import COCODemo

def main():
    if getattr(sys, 'frozen', False):
        path = Path(sys._MEIPASS)
    else:
        path = Path(__file__).parent
    args, cfg_fcos = fcos_cfgs(path)
    folder_list = [
     'results_det', 'images_rec']
    for folder in folder_list:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tiff', '.TIFF']
    print(f"Supported image extensions {extensions}")
    images_path = os.path.join(os.getcwd(), 'images_det')
    image_name_list = sorted([name for name in os.listdir(images_path) if os.path.splitext(name)[1] in extensions])
    coco_demo = COCODemo(cfg_fcos,
      confidence_thresholds_for_classes=(args.thresholds_for_classes),
      min_image_size=(args.min_image_size))
    print('##### Demo Starts #####')
    for idx, im_name in enumerate(image_name_list):
        start_time = time.time()
        image = cv2.imread(os.path.join(images_path, im_name))
        result = coco_demo.run_on_opencv_image(im_name, image)
        cv2.imwrite(f"results_det/{im_name}", result)
        print(f"{im_name}\tTime:{time.time() - start_time:.4f}")

    coco_demo.save_cropped_names()
    print('Finished.')


if __name__ == '__main__':
    main()
# okay decompiling run.pyc
