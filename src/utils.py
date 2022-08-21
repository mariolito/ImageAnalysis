import cv2
scale_percent = 0.5
img_size_thresh = 700
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


def resize(img):
    height, width, _ = img.shape
    while (width > img_size_thresh) | (height > img_size_thresh):
        width = int(width * scale_percent )
        height = int(height * scale_percent)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        height, width, _ = img.shape
    return img


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
