import cv2
import numpy as np
scale = 0.00392
# read class names from text file
classes = None
import os

with open(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'yolo', 'coco.names'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
# read pre-trained model and config file
net = cv2.dnn.readNet(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'yolo', 'saved_models', 'yolov3.weights'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'yolo', 'yolo3.cfg')
)


def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = []

    for i in net.getUnconnectedOutLayers():
        output_layers.append(layer_names[i - 1])
    return output_layers


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def detect_objects_yolo(img):
    Width = img.shape[1]
    Height = img.shape[0]
    # create input blob
    blob = cv2.dnn.blobFromImage(img, scale, (416, 416), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    result = {
        'prediction_class': [],
        'prediction_box': [],
        'confidence': []
    }
    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        result['prediction_class'].append(classes[class_ids[i]])
        result['prediction_box'].append(box)
        result['confidence'].append(confidences[i])
        draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    result['img'] = img
    return result
