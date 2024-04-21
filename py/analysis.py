import numpy as np
import cv2 as cv
import logging
import argparse
from src.utils import load_paired_images
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script for analyzing depth map')
    parser.add_argument("--parameters_path", type=str, required=True,
                        help="path to file with parameters")
    parser.add_argument("--photos_path", type=str, required=True,
                        help="path to directory with test photos")
    args = parser.parse_args()
    cv_file = cv.FileStorage(args.parameters_path, cv.FILE_STORAGE_READ)
    baseline = cv_file.getNode("Baseline").mat()[0][0]
    left_cam_mat = cv_file.getNode("Left_cam_mat").mat()
    right_cam_mat = cv_file.getNode("Right_cam_mat").mat()
    cv_file.release()

    images = load_paired_images(args.photos_path)
    if not images:
        logging.error("path don't has valid photos")
        exit()

    f_pixel = left_cam_mat[0][0]

    stereo = cv.StereoSGBM.create()
    for imgL, imgR in images:
        disparity = stereo.compute(imgL, imgR)
        mean = np.mean(disparity[disparity != 0])
        disparity[disparity == 0] = mean

        if baseline < 0:
            baseline *= -1

        disparity[disparity < 0] *= -1
        norm_disparity = (disparity.astype(np.float32)/16)
        z = (baseline * f_pixel) / norm_disparity
        plt.imshow(np.hstack((imgL, imgR)))
        plt.show()
        plt.imshow(z, cmap='inferno')
        plt.colorbar()
        plt.show()
        cv.waitKey(0)

        plt.savefig('depth_map.png')
        plt.close()
        depth_map = cv.imread('depth_map.jpg')

        print(np.unique(z, return_counts=True))
        # blended_image = cv.addWeighted(imgL, 0.6, depth_map, 0.4, 0)
        # cv.imshow('Blended Image', blended_image)

        if cv.waitKey() == 27:
            np.set_printoptions(threshold=np.inf)
            file = open('depth.txt', 'w')
            file.write(np.array2string(z, max_line_width=None))
            file.close()
            file = open('disparity.txt', 'w')
            file.write(np.array2string(norm_disparity, max_line_width=None))
            file.close()

    cv.destroyAllWindows()


