import cv2


def calculate_color_metrics(image_info):
    segmented_img = image_info['segment_utils_results']['segmented']
    background = 255 - segmented_img
    foreground = segmented_img
    norm_image = cv2.normalize(image_info['img'], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    back_avg_norm, back_std_norm = cv2.meanStdDev(norm_image, mask=background)
    fore_avg_norm, fore_std_norm = cv2.meanStdDev(norm_image, mask=foreground)
    back_avg, back_std = cv2.meanStdDev(image_info['img'], mask=background)
    fore_avg, fore_std = cv2.meanStdDev(image_info['img'], mask=foreground)
    result = {
        'fore_avg': fore_avg, 'fore_std': fore_std, 'back_avg': back_avg, 'back_std': back_std,
        'foreground': foreground, 'background': background, 'back_avg_norm': back_avg_norm, 'back_std_norm': back_std_norm,
        'fore_avg_norm': fore_avg_norm, 'fore_std_norm': fore_std_norm
    }
    return result
