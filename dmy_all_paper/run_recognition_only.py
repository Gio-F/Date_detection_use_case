# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: run.py
import json, os, cv2, time, sys
from pathlib import Path
from DMY.dt_and_rec import DetectRecognizeDmy
from DMY.dmy_utils import load_networks

def main():
    if getattr(sys, 'frozen', False):
        path = Path(sys._MEIPASS)
    else:
        path = Path(__file__).parent
    models = load_networks(path)
    detect_rec_dmy = DetectRecognizeDmy(models)
    read_from_json = False
    images_path = os.path.join(os.getcwd(), 'images_rec')
    images_path = os.path.join(os.getcwd(), 'images_rec')
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tiff', '.TIFF']
    try:
        with open('images_rec/cropped_img_list.json') as (f):
            data = json.load(f)
            image_name_list = [k for k, v in data.items()]
            read_from_json = True
    except:
        print('There is no cropped_img_list.json file.')
        image_name_list = sorted([name for name in os.listdir(images_path) if os.path.splitext(name)[1] in extensions])

    if not os.path.isdir('results_rec'):
        os.makedirs('results_rec')
    recognition = []
    print('##### Demo Starts #####')
    for im_name in image_name_list:
        images = []
        if read_from_json:
            for name in data[im_name]:
                images.append(cv2.imread(os.path.join(images_path, name)))

        else:
            images.append(cv2.imread(os.path.join(images_path, im_name)))
        start_time = time.time()
        det_date, rec_date, labels = detect_rec_dmy(images)
        if det_date is not None:
            recognition.append(f"{im_name}: {rec_date}\tLabels: {labels}")
            cv2.imwrite(f"results_rec/{im_name}", det_date)
        print(f"{im_name}\tTime:{time.time() - start_time:.2f}s\tPredicted Date: {rec_date}\tLabels:{labels}")

    with open('results_rec/recognized dates.txt', 'w') as (f):
        for r in recognition:
            f.writelines(f"{r}\n")

    print('Finished.')


if __name__ == '__main__':
    main()
# okay decompiling run.pyc
