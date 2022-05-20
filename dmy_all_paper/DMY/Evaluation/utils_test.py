# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/ExpDateRecognition/DMY/Evaluation/utils_test.py
# Compiled at: 2021-12-17 12:32:50
# Size of source mod 2**32: 15568 bytes
import torch, torchvision
from torchvision import transforms
import numpy as np, cv2, datetime
from DMY.Evaluation.bounding_box import BoxList
from PIL import Image
CATEGORIES = [
 'background', 'day', 'month', 'year']

def compute_locations(features):
    stride = [
     2, 4, 8]
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        locations_per_level = compute_locations_per_level(h, w, stride[level], feature.device)
        locations.append(locations_per_level)

    return locations


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(0, (w * stride), step=stride, dtype=(torch.float32), device=device)
    shifts_y = torch.arange(0, (h * stride), step=stride, dtype=(torch.float32), device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def sort_predictions(predictions):
    """
    Sort prediction according to their scores.
    """
    scores = predictions.get_field('scores')
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def overlay_boxes(image, predictions, boxes):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        orig_img: an image as returned by OpenCV
        image: detected expiration date region
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
        dates: recognized day, month, and year
        exp_pred: top-left corner coordinate of the detected expiration date box
    Returns:
        orig_img: detection and recognition results on original image on the image as returned by OpenCV
        labels: day, month, and year labels for recognition results
    """
    colors = {'year':[
      0, 0, 255], 
     'month':[255, 255, 0],  'day':[0, 255, 0]}
    for key, value in predictions.items():
        color = colors[key]
        x1, y1, x2, y2 = boxes[key]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image


def select_top_one_per_class(top_predictions, size):
    """
     Selects a prediction with highest scores for each class (day, month, and year)
    """
    boxes, labels, scores = check_overlap_select_2nd(top_predictions)
    index = boxes[:, 0].sort()[1]
    boxes = boxes[index]
    labels = labels[index]
    scores = scores[index]
    top_predictions = BoxList(boxes, size, mode='xyxy')
    boxlist = BoxList(boxes, (top_predictions.size), mode='xyxy')
    boxlist.add_field('labels', labels)
    boxlist.add_field('scores', scores)
    return boxlist


def pick_highest_ones(labels, scores, boxes):
    pick_labels = []
    pick_scores = []
    pick_boxes = []
    for l in torch.unique(labels):
        h_s = torch.max((scores[(l == labels)]), dim=0)[0]
        idx = (scores == h_s).nonzero()
        pick_scores.append(h_s.unsqueeze(0))
        pick_labels.append(labels[idx].unsqueeze(0))
        pick_boxes.append(boxes[idx].unsqueeze(0))

    pick_scores = torch.cat(pick_scores).squeeze()
    pick_labels = torch.cat(pick_labels).squeeze()
    pick_boxes = torch.cat(pick_boxes).squeeze()
    return (
     pick_labels, pick_scores, pick_boxes)


def check_overlapping(boxes):
    iou = box_iou(boxes, boxes)
    iou = torch.triu(iou, diagonal=1)
    overlapped = []
    if iou[(0, 1)] > 0.3:
        overlapped.append([0, 1])
    if len(boxes) == 3:
        if iou[(0, 2)] > 0.3:
            overlapped.append([0, 2])
        if iou[(1, 2)] > 0.3:
            overlapped.append([1, 2])
    return overlapped


def pick_changing_label(overlapped, labels, scores):
    lab1 = overlapped[0][0] + 1
    lab2 = overlapped[0][1] + 1
    if lab1 in labels:
        if lab2 in labels:
            idx1 = lab1 == labels
            idx2 = lab2 == labels
            if scores[idx1] > scores[idx2]:
                changing_label = lab2
            else:
                changing_label = lab1
    else:
        if lab1 in labels:
            changing_label = lab1
        else:
            changing_label = lab2
    return changing_label


def pick_2nd_highest_ones(changing_label, labels, scores, boxes):
    h_s = torch.sort((scores[(changing_label == labels)]), 0, descending=True)[0][1]
    idx = (scores == h_s).nonzero().squeeze(0)
    return (labels[idx], scores[idx], boxes[idx])


def change_prediction(h_labels, h_scores, h_boxes, change_label, labels, scores, boxes):
    new_l, new_s, new_b = pick_2nd_highest_ones(change_label, labels, scores, boxes)
    h_scores[h_labels == change_label] = new_s
    h_boxes[h_labels == change_label] = new_b
    change_label_score = torch.max((scores[(labels == change_label)]), dim=0)[0]
    idx_del = scores != change_label_score
    labels = labels[idx_del]
    scores = scores[idx_del]
    boxes = boxes[idx_del]
    overlapped_2 = check_overlapping(h_boxes)
    if len(overlapped_2) > 0:
        change_label = pick_changing_label(overlapped_2, h_labels, h_scores)
        if (change_label == labels).sum() > 1:
            new_l, new_s, new_b = pick_2nd_highest_ones(change_label, labels, scores, boxes)
            h_scores[h_labels == change_label] = new_s
            h_boxes[h_labels == change_label] = new_b
        else:
            h_labels, h_scores, h_boxes = remove_overlapped_prediction(h_labels, h_scores, h_boxes, change_label)
    return (
     h_labels, h_scores, h_boxes)


def remove_overlapped_prediction(h_labels, h_scores, h_boxes, change_label):
    del_idx = h_labels != change_label
    h_labels = h_labels[del_idx]
    h_scores = h_scores[del_idx]
    h_boxes = h_boxes[del_idx]
    return (h_labels, h_scores, h_boxes)


def check_overlap_select_2nd(prd):
    labels = prd.get_field('labels')
    scores = prd.get_field('scores')
    boxes = prd.bbox
    h_labels, h_scores, h_boxes = pick_highest_ones(labels, scores, boxes)
    if h_labels.numel() > 1:
        overlapped = check_overlapping(h_boxes)
        if len(overlapped) > 0:
            change_label = pick_changing_label(overlapped, h_labels, h_scores)
            if (change_label == labels).sum() > 1:
                h_labels, h_scores, h_boxes = change_prediction(h_labels, h_scores, h_boxes, change_label, labels, scores, boxes)
            else:
                h_labels, h_scores, h_boxes = remove_overlapped_prediction(h_labels, h_scores, h_boxes, change_label)
    else:
        h_labels = h_labels.unsqueeze(0)
        h_scores = h_scores.unsqueeze(0)
        h_boxes = h_boxes.unsqueeze(0)
    return (h_boxes, h_labels, h_scores)


def crop_boxes(img, top_predictions, transforms):
    """
    Crops detected day, month, and year boxes

    Arguments:
        img: detected expiration date region
        top_predictions: day, month, and year predictions
    Return:
         crop_imgs: cropped day, month, and year region
    """
    m = 2
    boxes = top_predictions.bbox
    h, w, _ = img.shape
    crop_imgs = []
    for idx_2, box in enumerate(boxes):
        box = box.to(torch.int64)
        x1, y1, x2, y2 = box.tolist()
        if x1 <= 0 or y1 <= 0:
            img_cropped = img[y1:y2 + m, x1:x2 + m]
        else:
            img_cropped = img[max(0, y1 - m):min(y2 + m, h - 1), max(0, x1 - m):min(x2 + m, w - 1)]
        img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
        img_cropped = Image.fromarray(img_cropped).convert('L')
        img_cropped = transforms(img_cropped)
        crop_imgs.append(img_cropped)

    return torch.stack(crop_imgs, dim=0)


def compare_years(temp_date, temp_img, temp_box, date, k, images, boxes):
    if date['year'] > temp_date['year']:
        temp_date = date
        temp_img, temp_box = change_img_and_box(k, images, boxes)
    else:
        if date['year'] == temp_date['year']:
            if 'month' in temp_date:
                if 'month' in date:
                    temp_date, temp_img, temp_box = compare_months(temp_date, temp_img, temp_box, date, k, images, boxes)
    return (
     temp_date, temp_img, temp_box)


def compare_months(temp_date, temp_img, temp_box, date, k, images, boxes):
    temp_month, date_month = digit_months(temp_date, date)
    if date_month > temp_month:
        temp_date = date
        temp_img, temp_box = change_img_and_box(k, images, boxes)
    else:
        if date['month'] == temp_date['month'] and 'day' in date and 'day' in temp_date and 'day' in temp_date:
            if 'day' in date:
                temp_date, temp_img, temp_box = compare_days(temp_date, temp_img, temp_box, date, k, images, boxes)
    return (
     temp_date, temp_img, temp_box)


def compare_days(temp_date, temp_img, temp_box, date, k, images, boxes):
    if date['day'] > temp_date['day']:
        temp_date = date
        temp_img, temp_box = change_img_and_box(k, images, boxes)
    return (
     temp_date, temp_img, temp_box)


def change_img_and_box(k, images, boxes):
    temp_img = images[k]
    temp_box = boxes[k]
    return (temp_img, temp_box)


def digit_months(temp_date, date):
    digit_months = {'JAN':1, 
     'FEB':2,  'MAR':3,  'APR':4,  'MAY':5,  'JUN':6,  'JUL':7, 
     'AUG':8,  'SEP':9,  'OCT':10,  'NOV':11,  'DEC':12}
    if date['month'].isalpha():
        date_month = digit_months[date['month']]
    else:
        date_month = date['month']
    if temp_date['month'].isalpha():
        temp_month = digit_months[temp_date['month']]
    else:
        temp_month = temp_date['month']
    return (temp_month, date_month)


def select_expiration_date(imgs, prds, bboxes):
    images, dates, boxes = check_date_in_suitable_format(imgs, prds, bboxes)
    if len(dates) > 1:
        init_key = [
         *images][0]
        temp_img = images.pop(init_key)
        temp_date = dates.pop(init_key)
        temp_box = boxes.pop(init_key)
        for k, date in dates.items():
            if len(temp_date) == len(date):
                if 'year' in temp_date and 'year' in date:
                    temp_date, temp_img, temp_box = compare_years(temp_date, temp_img, temp_box, date, k, images, boxes)
            elif 'month' in temp_date:
                if 'month' in date:
                    temp_date, temp_img, temp_box = compare_months(temp_date, temp_img, temp_box, date, k, images, boxes)
            else:
                if 'day' in temp_date:
                    if 'day' in date:
                        temp_date, temp_img, temp_box = compare_days(temp_date, temp_img, temp_box, date, k, images, boxes)
                    if 'day' in date and 'month' in date:
                        temp_date = date
                        temp_img, temp_box = change_img_and_box(k, images, boxes)
                else:
                    if 'year' in temp_date:
                        if 'year' in date:
                            temp_date, temp_img, temp_box = compare_years(temp_date, temp_img, temp_box, date, k, images, boxes)
                    if 'year' in date:
                        temp_date = dates[k]
                        temp_img, temp_box = change_img_and_box(k, images, boxes)
                    else:
                        if 'year' in temp_date:
                            continue
                        if 'month' in temp_date:
                            if 'month' in date:
                                temp_date, temp_img, temp_box = compare_months(temp_date, temp_img, temp_box, date, k, images, boxes)
                        if 'month' in date:
                            temp_date = dates[k]
                            temp_img, temp_box = change_img_and_box(k, images, boxes)
                        else:
                            if 'month' in temp_date:
                                continue
                            print('bbb')

        return (
         temp_img, temp_date, temp_box)
    else:
        if len(dates) == 1:
            for k, v in dates.items():
                key = k

            return (
             images[key], dates[key], boxes[key])
        return (None, None, None)


def merge_date_labels(exp_date):
    date, labels = [], []
    for k, v in exp_date.items():
        labels.append(k)
        date.append(v)

    return (' '.join(date), ','.join(labels))


def check_date_in_suitable_format(images, dates, boxes):
    months = [
     'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    delete = []
    for k, v in dates.items():
        if 'day' in v:
            if not v['day'].isdigit() or not int(v['day']) < 32 or not len(v['day']) == 2:
                delete.append(k)
            if 'month' in v:
                if v['month'].isdigit():
                    if not int(v['month']) < 13 or not (len(v['month']) == 2 or len(v['month']) == 3):
                        delete.append(k)
                elif v['month'] not in months or not (len(v['month']) == 2 or len(v['month']) == 3):
                    delete.append(k)
            if 'year' in v and (not v['year'].isdigit() or not (len(v['year']) == 2 or len(v['year']) == 4)):
                delete.append(k)

    for d in set(delete):
        del dates[d]
        del images[d]
        del boxes[d]

    return (images, dates, boxes)


def date_dictionary(predictions, dates):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        orig_img: an image as returned by OpenCV
        image: detected expiration date region
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
        dates: recognized day, month, and year
        exp_pred: top-left corner coordinate of the detected expiration date box
    Returns:
        orig_img: detection and recognition results on original image on the image as returned by OpenCV
        labels: day, month, and year labels for recognition results
    """
    labels = predictions.get_field('labels')
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions.bbox.int().tolist()
    dates = dates.split(' ')
    dict_date, dict_box = {}, {}
    for idx, (box, label, date) in enumerate(zip(boxes, labels, dates)):
        dict_date[label] = date
        dict_box[label] = box

    return (
     dict_date, dict_box)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x0, y0, x1, y1) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x0, y0, x1, y1) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou
# okay decompiling ./utils_test.pyc
