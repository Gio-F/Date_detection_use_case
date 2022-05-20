# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/ExpDateDetection/FCOS/fcos_core/predictor.py
# Compiled at: 2021-12-17 01:35:52
# Size of source mod 2**32: 8361 bytes
import cv2, torch, numpy as np, os, json
from torchvision import transforms as T
from FCOS.fcos_core.modeling.detector import build_detection_model
from FCOS.fcos_core.utils.checkpoint import DetectronCheckpointer
from FCOS.fcos_core.structures.image_list import to_image_list

class COCODemo(object):

    def __init__(self, cfg, confidence_thresholds_for_classes, min_image_size=224):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size
        checkpointer = DetectronCheckpointer(cfg, (self.model), save_dir=(cfg.OUTPUT_DIR))
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        self.transforms = self.build_transform()
        self.cpu_device = torch.device('cpu')
        self.confidence_thresholds_for_classes = torch.tensor(confidence_thresholds_for_classes)
        self.categories = [
         'background',
         'code',
         'due',
         'exp',
         'prod']
        self.colors = {'code':(255, 0, 0), 
         'due':(0, 128, 255), 
         'exp':(0, 255, 0), 
         'prod':(128, 0, 128)}
        self.collect_cropped_names = {}

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
        normalize_transform = T.Normalize(mean=(cfg.INPUT.PIXEL_MEAN),
          std=(cfg.INPUT.PIXEL_STD))
        transform = T.Compose([
         T.ToPILImage(),
         T.Resize(self.min_image_size),
         T.ToTensor(),
         to_bgr_transform,
         normalize_transform])
        return transform

    def crop_boxes(self, img_name, img, boxes, labels):
        """
        Crops expiration date candidate region.
        """
        name_list = []
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            if label == 3:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                tb, lr = (5, 5)
                if x1 <= lr or y1 <= tb:
                    cropped_img = img[y1:y2 + tb, x1:x2 + lr]
                else:
                    cropped_img = img[y1 - tb:y2 + tb, x1 - lr:x2 + lr]
                if cropped_img is not None:
                    name, ext = os.path.splitext(img_name)
                    cv2.imwrite(f"images_rec/{name}_{idx:02}{ext}", cropped_img)
                    name_list.append(f"{name}_{idx:02}{ext}")
            self.collect_cropped_names[img_name] = name_list

    def run_on_opencv_image(self, name, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        self.crop_boxes(name, image, top_predictions.bbox, top_predictions.extra_fields['labels'])
        image = self.overlay_boxes(image, top_predictions)
        image = self.overlay_class_names(image, top_predictions)
        return image

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions[0]]
        prediction = predictions[0]
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field('scores')
        labels = predictions.get_field('labels')
        thresholds = self.confidence_thresholds_for_classes[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field('scores')
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field('labels')
        labels = [self.categories[i] for i in labels]
        boxes = predictions.bbox
        for box, label in zip(boxes, labels):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            color = self.colors[label]
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 2)

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field('scores').tolist()
        labels = predictions.get_field('labels').tolist()
        labels = [self.categories[i] for i in labels]
        boxes = predictions.bbox
        size = 0.8
        thick = 2
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2].numpy().astype('int')
            s = '{}: {:.2f}'.format(label, score)
            color = self.colors[label]
            cv2.putText(image, s, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)

        return image

    def save_cropped_names(self):
        with open('images_rec/cropped_img_list.json', 'w') as (f):
            json.dump((self.collect_cropped_names), f, indent=4)
# okay decompiling ./predictor.pyc
