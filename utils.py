import cv2
import numpy as np
import random
import os
import dlib

from sklearn.utils import shuffle
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, JpegCompression
)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../Data/shape_predictor_68_face_landmarks.dat")
face_detector = dlib.get_frontal_face_detector()

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom


def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]

def get_head(img):
    det = detector(img, 1)[0]
    height, width = img.shape[:2]

    lr = det.right() - det.left()
    tb = det.bottom() - det.top()
    det = dlib.rectangle(left=max(0, det.left() - int(0.25 * lr)),
                         top=max(0, det.top() - int(0.5 * tb)),
                         right=min(width, det.right() + int(0.25 * lr)),
                         bottom=min(height, det.bottom() + int(0.25 * tb)))
    cropped = crop_image(img, det)
    return cropped


def load_data_test_generator(x, batch_size=64):
    num_samples = x.shape[0]
    while 1:
        for i in range(0, num_samples, batch_size):
            x_data = [preprocess_image(im) for im in x[i:i + batch_size]]
            yield np.array(x_data)


def load_data_generator(x, y, batch_size=64):
    num_samples = x.shape[0]
    while 1:
        try:
            for i in range(0, num_samples, batch_size):
                x_data = [preprocess_image(im) for im in x[i:i + batch_size]]
                y_data = y[i:i + batch_size]

                yield shuffle(np.array(x_data), np.array(y_data))
        except Exception as err:
            print(err)


def face_aug(p=.5):
    return Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(scale=(1, 3)),
            GaussNoise(var_limit=(1, 5)),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(alpha=(0.1, 0.2)),
            IAAEmboss(strength=(0.1, 0.3)),
            RandomContrast(limit=0.1),
            RandomBrightness(limit=0.15),
        ], p=0.3)
    ], p=p)

def augmentation(image):
    aug = face_aug(p=1)
    image = aug(image=image)['image']
    return image


def preprocess_celeba(attr_path):
    lines = [line.rstrip() for line in open(attr_path, 'r')]
    all_attr_names = lines[1].split()
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name
    lines = lines[2:]
    with_g = []
    without_g = []
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]
        idx = attr2idx['Eyeglasses']
        if values[idx] == '1':
            with_g.append(filename)
        else:
            without_g.append(filename)
    return with_g, without_g


def preprocess_image(image_path, size=128):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (size, size))
    img = img/255.0
    return img


def str2bool(x):
    return x.lower() in ('true')


def prepare_data_dir(path='./dataset'):
    if not os.path.exists(path):
        os.makedirs(path)
