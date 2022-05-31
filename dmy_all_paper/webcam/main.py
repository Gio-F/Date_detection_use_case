# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22)
# [GCC 10.2.1 20210110]
# Embedded file name: webcam.py
import cv2, torch, time, sys
from pathlib import Path
from FCOS.configs.configs import fcos_cfgs
from FCOS.fcos_core.predictor import COCODemo
from DMY.dt_and_rec import DetectRecognizeDmy
from DMY.dmy_utils import load_networks

def main(path_to_picture):
    if getattr(sys, 'frozen', False):
        path = Path(sys._MEIPASS)
        print(path)
    else:
        path = Path(__file__).parent
        print(path)
    num_gpus = torch.cuda.device_count()
    print('# of GPUs: {}'.format(num_gpus))
    args, cfg_fcos = fcos_cfgs(path)
    models = load_networks(path)
    detect_rec_dmy = DetectRecognizeDmy(models)
    coco_demo = COCODemo(cfg_fcos,
      confidence_thresholds_for_classes=(args.thresholds_for_classes),
      min_image_size=(args.min_image_size))
    image = cv2.imread(path_to_picture)
    start_time=time.time()
    image, date_images, tl_info = coco_demo.run_on_opencv_image(image)
    if date_images is not None:
        image, rec_date, _ = detect_rec_dmy(image, date_images, tl_info)
        t = time.time()-start_time
        return rec_date, t
    else:
        return "The date cannot be detected.", time.time()-start_time


if __name__ == '__main__':
    path_pic = sys.argv[1]
    result, t = main(path_pic)
    print(result)
    print(f"In {t} s")
# okay decompiling webcam.pyc
