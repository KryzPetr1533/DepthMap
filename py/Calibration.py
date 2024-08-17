import cv2 as cv
import numpy as np
import glob


def calibration():
    # parameters
    chessboardSize = (6, 9)
    cellSize = 4.0  # size of the chessboard cell in cm

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare data structure
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []
    imgpointsL = []
    imgpointsR = []

    # Paths to photos for the calibration
    path = ('C:\\Users\\petra\\OneDrive\\Documents\\My_docs\\Drone\\DepthMap\\CameraCalibTestPhotos\\camera_calib_6x9\\')

    extention = ".jpg"
    imageL = "imgL"
    imageR = "imgR"
    imagesLeft = glob.glob(path + imageL + '*' + extention).sort()
    imagesRight = glob.glob(path + imageR + '*' + extention).sort()

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):
        imgL = cv.imread(imgLeft)
        imgR = cv.imread(imgRight)
        outputL = imgL.copy()
        outputR = imgR.copy()
        grayL = cv.cvtColor(outputL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(outputR, cv.COLOR_BGR2GRAY)
        retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None,
                                                  cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None,
                                                  cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        if retR and retL:
            objpoints.append(objp)
            cv.cornerSubPix(grayR,
                            cornersR,
                            (2 * min(chessboardSize[0], chessboardSize[1]) - 1,
                             2 * min(chessboardSize[0], chessboardSize[1]) - 1),
                            (-1, -1),
                            criteria)
            cv.cornerSubPix(grayL,
                            cornersL,
                            (2 * min(chessboardSize[0], chessboardSize[1]) - 1,
                             2 * min(chessboardSize[0], chessboardSize[1]) - 1),
                            (-1, -1),
                            criteria)
            cv.drawChessboardCorners(outputR, chessboardSize, cornersR, retR)
            cv.drawChessboardCorners(outputL, chessboardSize, cornersL, retL)
            cv.imshow('corners', np.hstack((outputR, outputL)))
            cv.waitKey(0)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
    cv.destroyAllWindows()

    print("Calculating left camera parameters ... ")
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    hL, wL = grayL.shape[:2]
    new_mtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    print("Calculating right camera parameters ... ")
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
    hR, wR = grayR.shape[:2]
    new_mtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

    print("Stereo calibration .....")
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv.stereoCalibrate(objpoints,
                                                                                       imgpointsL,
                                                                                       imgpointsR,
                                                                                       new_mtxL,
                                                                                       distL,
                                                                                       new_mtxR,
                                                                                       distR,
                                                                                       grayL.shape[::-1],
                                                                                       criteria_stereo,
                                                                                       flags)

    baseline = Trns[0] * cellSize
    rectify_scale = 1  # if 0 image croped, if 1 image not croped
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                                             grayL.shape[::-1], Rot, Trns,
                                                                             rectify_scale, (0, 0))

    Left_Stereo_Map = cv.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                                 grayL.shape[::-1], cv.CV_16SC2)
    Right_Stereo_Map = cv.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                                  grayR.shape[::-1], cv.CV_16SC2)
    print("Baseline: ", baseline)
    print("Left_camera_matrix:", mtxL)
    print("Right_camera_matrix:", mtxR)
    print("Saving parameters ......")
    cv_file = cv.FileStorage(".\\dataparamspy.xml", cv.FILE_STORAGE_WRITE)
    cv_file.write("Left_cam_mat", mtxL)
    cv_file.write("Right_cam_mat", mtxR)
    cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
    cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
    cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
    cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
    cv_file.write("Trns", Trns)
    cv_file.write("Baseline", baseline)
    cv_file.release()
    print("Finishing Calibration ...")


calibration()
