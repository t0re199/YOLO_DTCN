import os
import sys
import cv2
import json
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from optparse import OptionParser

from yolo.models import Darknet, load_darknet_weights
from yolo.utils.utils import non_max_suppression
from transforms.YoloTransform import build_image_transformer


__YOLO_DIR__ = "../.yolov3"
__YOLO_IMG_SIZE__ = 416

__COCO_CLASSES_DIR__ = "../data/coco_classes.json"


coco_classes = None
class_colors = None

device = "cpu"
yolo = None
image_transformer = None

user_param = {}


def build_yolo(path=__YOLO_DIR__, image_size=__YOLO_IMG_SIZE__):
    yolo_model = Darknet(os.path.join(path, "cfg/yolov3-tiny.cfg"), image_size).to(device)
    load_darknet_weights(yolo_model, os.path.join(path, "weights/yolov3-tiny.weights"))
    return yolo_model


def load_coco_classes(path=__COCO_CLASSES_DIR__):
    with open(path, "r") as fd:
        return json.load(fd)


def __init():
    coco_classes = load_coco_classes()
    class_colors = np.random.randint(0x0, 0x100, size=(len(coco_classes), 0x3)) / 255.0

    yolo = build_yolo()
    yolo.eval()

    image_transformer = build_image_transformer(__YOLO_IMG_SIZE__)


def annotate_image(tensor, classes_, bboxes_, confs_):
    output_img = cv2.cvtColor(tensor[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)

    for i in range(len(classes_)):
        if confs_[i] > user_param["confidence"]:
            x, y, w, h = bboxes_[i]
            color = class_colors[classes_[i]]
            cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 0x3)
            object_class = coco_classes[classes_[i]]

            annotation = f"{object_class}: {round(confs_[i], 0x2)}"
            print(annotation, bboxes_[i])
            cv2.putText(output_img, annotation, (x, max(5, y - 0x5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 0x2)


def detect(data):
    with torch.no_grad():
        detections = yolo(data).cpu()

    detections = non_max_suppression(detections, user_param["confidence"], user_param["overlap"])[0x0]

    classes = []
    bboxes = []
    confidences = []

    for x1, y1, x2, y2, _, cls_conf, cls_pred in detections:
        box_h = y2 - y1
        box_w = x2 - x1

        y1 = y1 // 0x2
        x1 = x1 // 0x2

        bboxes.append((int(x1), int(y1), int(box_w), int(box_h)))
        confidences.append(float(cls_conf.item()))
        classes.append(int(cls_pred))

    return annotate_image(data, classes, bboxes, confidences)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--image", dest="image")
    parser.add_option("-c", "--confidence", dest="confidence")
    parser.add_option("-o", "--overlap", dest="overlap")
    parser.add_option("-O", "--output", dest="output")
    parser.add_option("-C", "--cuda", action="store_true", dest="cuda")
    (options, args) = parser.parse_args()

    if options.image is None or not os.path.isfile(options.image):
        sys.stderr.write("Invalid Input\n")
        exit(0x1)

    user_param["image"] = options.image

    if options.confidence is None:
        sys.stderr.write("Invalid Confidences\n")
        exit(0x2)

    try:
        user_param["confidence"] = float(options.confidence)
    except ValueError:
        sys.stderr.write("Invalid Confidences\n")
        exit(0x2)

    if options.overlap is None:
        sys.stderr.write("Invalid Overlap\n")
        exit(0x2)

    try:
        user_param["overlap"] = float(options.confidence)
    except ValueError:
        sys.stderr.write("Invalid Overlap\n")
        exit(0x3)

    user_param["output"] = options.output

    device = "cuda" if options.cuda else "cpu"
    __init()

    pil_image = Image.open(user_param["image"])
    tensor = image_transformer(pil_image).unsqueeze(0x0)

    annotated_image = detect(tensor)

    if user_param["output"] is None:
        plt.figure(figsize=[0xa] * 0x2)
        plt.axis(False)
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        cv2.imwrite(user_param["output"], annotated_image)