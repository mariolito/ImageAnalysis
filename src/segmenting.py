import numpy as np
import cv2
from src.u2net.data_loader import RescaleTSingle, ToTensorLabSingle
from src.u2net.segmentation import segment
contor_percentage_covered_thresh = 0.015


def segment_img(image_info, prob_segmented=0.5):
    img = image_info['img']
    img_to_tensor = ToTensorLabSingle(flag=0)(img)
    predict_np = segment(img_to_tensor.reshape([1] + list(img_to_tensor.shape)))
    predict_np[predict_np < prob_segmented] = 0
    predict_np[predict_np >= prob_segmented] = 255
    segmented_u2net = predict_np.astype('uint8')
    result = {'segmented_u2net': segmented_u2net}
    return result


def _preprocess_box(box):
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    return np.array([box])


def create_single_contor_img(img, contor, contor_box):
    contor_mask = cv2.drawContours(np.zeros_like(img), [contor], -1, (255, 255, 255), cv2.FILLED, 1)
    contor_mask = contor_mask.astype(np.uint8)
    contor_mask = cv2.cvtColor(contor_mask, cv2.COLOR_BGR2GRAY)
    contor_img = cv2.bitwise_and(img, img, mask=contor_mask)
    contor_img[contor_mask==0] = [255, 255, 255]
    x1, y1, x2, y2 = _preprocess_box(contor_box)[0]
    contor_img = contor_img[y1:y2, x1:x2, :]
    return contor_img


def find_contors(image_info):
    segmented = image_info['segment_utils_results']['segmented']
    significant_contors_boxes, significant_contors = find_contor_on_segmented_img(segmented)
    image_info['segment_utils_results']['significant_contors_boxes'] = significant_contors_boxes
    image_info['segment_utils_results']['significant_contors'] = significant_contors
    image_info['segment_utils_results']['contor_imgs'] = []
    for contor, contor_box in zip(significant_contors, significant_contors_boxes):
        contor_img = create_single_contor_img(image_info['img'].copy(), contor, contor_box)
        image_info['segment_utils_results']['contor_imgs'].append(contor_img)
    return image_info


def find_contor_on_segmented_img(segmented):
    contours, hierarchy = cv2.findContours(segmented[:, :, np.newaxis], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contors_boxes = []
    significant_contors = []
    for cnt in contours:
        contor_percentage_covered = (cv2.contourArea(cnt) / (segmented.shape[0] * segmented.shape[1]))
        if contor_percentage_covered > contor_percentage_covered_thresh:
            box = np.array(cv2.boundingRect(cnt))
            significant_contors_boxes.append(box)
            significant_contors.append(cnt)
    return significant_contors_boxes, significant_contors


def segment_utils(image_info):
    segment_utils_results = segment_img(image_info, 0.95)
    segment_utils_results['segmented'] = segment_utils_results['segmented_u2net']
    segment_utils_results['segmentation_type'] = 'u2net'
    image_info['segment_utils_results'] = segment_utils_results
    image_info = find_contors(image_info)
    return image_info
