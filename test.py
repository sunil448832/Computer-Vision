from names import*
import cv2
import numpy as np
import torchvision
from torchvision import transforms as T


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def get_prediction(image, threshold):
    transform = T.Compose([T.ToTensor()])
    image = transform(image)
    pred = model([image])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i]
                  for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def get_object(image, obj, size, threshold=0.5):
    boxes, pred_cls = get_prediction(image, threshold)
    objects = []
    for i in range(len(boxes)):
        if pred_cls[i] == obj:
            y1, x1 = boxes[i][0]
            y2, x2 = boxes[i][1]
            img_cp = image[int(x1):int(x2), int(y1):int(y2)]
            img_cp = cv2.resize(img_cp, (size, size))
            objects.append(img_cp)
    return objects
