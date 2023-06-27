import cv2 as cv
import numpy as np
import glob

def calibration():
    # parameters
    chessboardSize = (9, 15)

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
    path = ".\\CameraCalibTest\\"
    extention = ".jpg"
    imageL = "imgL"
    imageR = "imgR"
    imagesLeft = glob.glob(path + imageL + '*' + extention)
    imagesRight = glob.glob(path + imageR + '*' + extention)

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):
        imgL = cv.imread(imgLeft)
        imgR = cv.imread(imgRight)
        outputL = imgL.copy()
        outputR = imgR.copy()
        grayL = cv.cvtColor(outputL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(outputR, cv.COLOR_BGR2GRAY)
        retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
        if retR and retL:
            objpoints.append(objp)
            cv.cornerSubPix(grayR, cornersR, (17, 17), (-1, -1), criteria)
            cv.cornerSubPix(grayL, cornersL, (17, 17), (-1, -1), criteria)
            cv.drawChessboardCorners(outputR, chessboardSize, cornersR, retR)
            cv.drawChessboardCorners(outputL, chessboardSize, cornersL, retL)
            cv.imshow('cornersR', outputR)
            cv.imshow('cornersL', outputL)
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

    rectify_scale = 1  # if 0 image croped, if 1 image not croped
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                                              grayL.shape[::-1], Rot, Trns,
                                                                              rectify_scale, (0, 0))

    Left_Stereo_Map = cv.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                                  grayL.shape[::-1], cv.CV_16SC2)
    Right_Stereo_Map = cv.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                                   grayR.shape[::-1], cv.CV_16SC2)
    print("Saving parameters ......")
    cv_file = cv.FileStorage(".\\dataparamspy.xml", cv.FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
    cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
    cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
    cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
    cv_file.write("Trns", Trns)
    cv_file.release()
    print("Finishing Calibration ...")