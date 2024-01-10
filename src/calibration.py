import logging
import argparse

import cv2 as cv
import numpy as np

from utils import load_paired_images


def stereo_calibration(paired_images, calibration_params_path, chessboard_size, cell_size, draw_images=False, verbose=False):

    min_size = min(chessboard_size[0], chessboard_size[1])
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_arr = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    obj_arr[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []
    img_points_l = []
    img_points_r = []

    gray_l = None
    gray_r = None
    for img_l, img_r in paired_images:
        output_l = img_l.copy()
        gray_l = cv.cvtColor(output_l, cv.COLOR_BGR2GRAY)

        output_r = img_r.copy()
        gray_r = cv.cvtColor(output_r, cv.COLOR_BGR2GRAY)

        if draw_images:
            cv.imshow("gray image left", gray_l)
            cv.imshow("gray image_right", gray_r)
            cv.waitKey(0)

        ret_l, corners_l = cv.findChessboardCorners(gray_l,
                                                    chessboard_size,
                                                    None,
                                                    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        ret_r, corners_r = cv.findChessboardCorners(gray_r,
                                                    chessboard_size,
                                                    None,
                                                    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret_r and ret_l:
            obj_points.append(obj_arr)
            cv.cornerSubPix(gray_l,
                            corners_l,
                            (2 * min_size - 1, 2 * min_size - 1),
                            (-1, -1),
                            criteria
                            )
            cv.cornerSubPix(gray_r,
                            corners_r,
                            (2 * min_size - 1, 2 * min_size - 1),
                            (-1, -1),
                            criteria
                            )
            cv.drawChessboardCorners(output_l, chessboard_size, corners_l, ret_l)
            cv.drawChessboardCorners(output_r, chessboard_size, corners_r, ret_r)

            img_points_l.append(corners_l)
            img_points_r.append(corners_r)

            if draw_images:
                cv.imshow('cornersL', output_l)
                cv.imshow('cornersR', output_r)
                cv.waitKey(0)

    cv.destroyAllWindows()

    if verbose:
        logging.info("Calculating left camera parameters ... ")
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(obj_points,
                                                           img_points_l,
                                                           gray_l.shape[::-1],
                                                           None,
                                                           None
                                                           )
    hL, wL = gray_l.shape[:2]
    new_mtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    if verbose:
        logging.info("Calculating right camera parameters ... ")
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(obj_points,
                                                           img_points_r,
                                                           gray_r.shape[::-1],
                                                           None,
                                                           None
                                                           )
    hR, wR = gray_r.shape[:2]
    new_mtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

    if verbose:
        logging.info("Stereo calibration .....")
    flags = cv.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv.stereoCalibrate(obj_points,
                                                                                       img_points_l,
                                                                                       img_points_r,
                                                                                       new_mtxL,
                                                                                       distL,
                                                                                       new_mtxR,
                                                                                       distR,
                                                                                       gray_l.shape[::-1],
                                                                                       criteria_stereo,
                                                                                       flags)

    baseline = Trns[0] * cell_size
    rectify_scale = 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                                             gray_l.shape[::-1], Rot, Trns,
                                                                             rectify_scale, (0, 0))

    Left_Stereo_Map = cv.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                                 gray_l.shape[::-1], cv.CV_16SC2)
    Right_Stereo_Map = cv.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                                  gray_r.shape[::-1], cv.CV_16SC2)
    if verbose:
        logging.info("Saving parameters ......")

    cv_file = cv.FileStorage(calibration_params_path, cv.FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
    cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
    cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
    cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
    cv_file.write("Trns", Trns)
    cv_file.write("Baseline", baseline)
    cv_file.release()

    if verbose:
        logging.info("Finishing Calibration ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script fot stereo camera calibration')
    parser.add_argument("--photos_path", type=str, required=True,
                        help="path to directory with paired images")
    parser.add_argument("--calibration_params_path", type=str, required=True,
                        help="path to directory with paired images")
    parser.add_argument('--chessboard_size', nargs=2, type=int, required=True,)
    parser.add_argument('--cell_size', type=int, required=True)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--draw', action='store_true')
    args = parser.parse_args()

    images = load_paired_images(args.photos_path)
    if not images:
        logging.error("path don't has valid photos")
        exit()

    stereo_calibration(images,
                       chessboard_size=args.chessboard_size,
                       cell_size=args.cell_size,
                       calibration_params_path=args.calibration_params_path,
                       draw_images=args.draw,
                       verbose=args.verbose)
