from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
import cv2
import numpy as np


class ImgFeaturesExtractor(object):

    def __init__(self, include_top=False, img_input_dim=299, normalize=False, black_and_white=False):
        self.model = Xception(include_top=include_top, weights='imagenet', classifier_activation=None)
        self.img_input_dim = img_input_dim
        self.normalize = normalize
        self.black_and_white = black_and_white

    def extract_features_multiple_imgs(self, images_info):
        x = np.zeros((len(images_info), self.img_input_dim, self.img_input_dim, 3))
        c = 0
        for i in images_info:
            x[c, :, :, :] = cv2.resize(images_info[i]['org_img'], (self.img_input_dim, self.img_input_dim),
                                        interpolation=cv2.INTER_AREA)
            c += 1
        x = preprocess_input(x)
        x = self.model.predict(x)
        output_vec_shape = 1
        for tmp_dim in x.shape[1:]:
            output_vec_shape *= tmp_dim

        x = np.reshape(x, (len(images_info), output_vec_shape))

        c = 0
        for i in images_info:
            images_info[i].update({'extracted_features': x[c, :].reshape(1, -1)})
        return images_info

    def extracted_features_of_contors(self, image_info):
        x = np.zeros((len(image_info['segment_utils_results']['contor_imgs']), self.img_input_dim, self.img_input_dim, 3))
        c = 0
        for contor_img in image_info['segment_utils_results']['contor_imgs']:
            x[c, :, :, :] = cv2.resize(contor_img, (self.img_input_dim, self.img_input_dim),
                                        interpolation=cv2.INTER_AREA)
            c += 1
        if len(image_info['segment_utils_results']['contor_imgs']) > 0:
            x = preprocess_input(x)
            x = self.model.predict(x)
            output_vec_shape = 1
            for tmp_dim in x.shape[1:]:
                output_vec_shape *= tmp_dim

            x = np.reshape(x, (len(image_info['segment_utils_results']['contor_imgs']), output_vec_shape))

        extracted_features_of_contor_imgs = {}
        for iii in range(len(image_info['segment_utils_results']['contor_imgs'])):
            extracted_features_of_contor_imgs[iii] = x[iii, :].reshape(1, -1)
        image_info['segment_utils_results']['extracted_features_of_contor_imgs'] = extracted_features_of_contor_imgs
        return image_info

    def extract(self, image):
        if self.black_and_white:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.img_input_dim, self.img_input_dim), interpolation=cv2.INTER_AREA)
        self.image = image

        # image = image.convert(mode='RGB')
        x = image
        # x = kimage.img_to_array(image)
        if self.normalize:
            x = (x - np.mean(x)) / (np.std(x))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = self.model.predict(x)
        output_vec_shape = 1
        for tmp_dim in x.shape:
            output_vec_shape *= tmp_dim
        return np.reshape(x, (1, output_vec_shape))
