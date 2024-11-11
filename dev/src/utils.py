import os
import glob
import cv2 as cv


def load_paired_images(path, left_suffix="imgL", right_suffix="imgR"):
    extention = ".jpg"

    left_images = glob.glob(os.path.join(path, left_suffix + '*' + extention))
    right_images = glob.glob(os.path.join(path, right_suffix + '*' + extention))

    images = []
    for left_image_path, right_image_path in zip(sorted(left_images), sorted(right_images)):
        left_image = cv.imread(left_image_path)
        right_image = cv.imread(right_image_path)
        images.append((left_image, right_image))

    return images
