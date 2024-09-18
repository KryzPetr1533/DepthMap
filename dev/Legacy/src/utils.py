import os
import glob
import cv2 as cv
import re


def load_paired_images(path, left_suffix="imgL", right_suffix="imgR"):
    # Function extracts the number from the name of the photo.
    def extract_number(filename):
        return int(re.search(r'(\d+)', filename).group())

    extention = ".jpg"

    left_images = glob.glob(os.path.join(path, left_suffix + '*' + extention))
    right_images = glob.glob(os.path.join(path, right_suffix + '*' + extention))

    right_images_dict = {extract_number(name): name for name in right_images}
    images = []
    for left_image_path in left_images:
        number = extract_number(left_image_path)
        right_image_path = right_images_dict.get(number)
        if right_image_path:
            left_image = cv.imread(left_image_path)
            right_image = cv.imread(right_image_path)
            print((right_image_path, left_image_path))
            images.append((left_image, right_image))

    return images
