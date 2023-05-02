import cv2
import numpy as np
def calibration():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9*15,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:15].T.reshape(-1,2)
    objpoints = []
    imgpointsL = []
    imgpointsR = []
    path = ".\\CameraCalibTest\\"
    extention = ".jpg"
    imageL = "imgL"
    imageR = "imgR"
    n = int(5)
    for i in range(0,n):
        nameL = path + imageL + str(i + 1) + extention
        nameR = path + imageR + str(i + 1) + extention
        imgL = cv2.imread(nameL)
        imgR = cv2.imread(nameR)
        outputL = imgL.copy()
        outputR = imgR.copy()
        grayL = cv2.cvtColor(outputL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(outputR, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(grayL, (9, 15), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (9, 15), None)
        if retR and retL:
            objpoints.append(objp)
            cv2.cornerSubPix(grayR, cornersR, (17, 17), (-1, -1), criteria)
            cv2.cornerSubPix(grayL, cornersL, (17, 17), (-1, -1), criteria)
            cv2.drawChessboardCorners(outputR, (9, 15), cornersR, retR)
            cv2.drawChessboardCorners(outputL, (9, 15), cornersL, retL)
            cv2.imshow('cornersR', outputR)
            cv2.imshow('cornersL', outputL)
            cv2.waitKey(0)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
    cv2.destroyAllWindows()
    print("Calculating left camera parameters ... ")
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,imgpointsL,grayL.shape[::-1],None,None)
    hL,wL= grayL.shape[:2]
    new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

    print("Calculating right camera parameters ... ")
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,imgpointsR,grayR.shape[::-1],None,None)
    hR,wR= grayR.shape[:2]
    new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

    print("Stereo calibration .....")
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(objpoints,
                                                              imgpointsL,
                                                              imgpointsR,
                                                              new_mtxL,
                                                              distL,
                                                              new_mtxR,
                                                              distR,
                                                              grayL.shape[::-1],
                                                              criteria_stereo,
                                                              flags)

    rectify_scale= 1 # if 0 image croped, if 1 image not croped
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                     grayL.shape[::-1], Rot, Trns,
                                                     rectify_scale,(0,0))


    Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                                 grayL.shape[::-1], cv2.CV_16SC2)
    Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                                  grayR.shape[::-1], cv2.CV_16SC2)
    print("Saving parameters ......")
    cv_file = cv2.FileStorage(".\\dataparamspy.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
    cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
    cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
    cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
    cv_file.write("Trns", Trns)
    cv_file.release()
    print("Finishing Calibration ...")