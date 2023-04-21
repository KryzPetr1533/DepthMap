import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
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
    cv_file = cv2.FileStorage(".Deepmap\\dataparamspy.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
    cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
    cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
    cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
    cv_file.write("Trns", Trns)
    cv_file.release()
    print("Finishing Calibration ...")

def nothing(x):
    pass



def ShowDisparity(imgLeft, imgRight, bSize=5):
    # Initialize the stereo block matching object
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=bSize)

    # Compute the disparity image
    disparity = stereo.compute(imgLeft, imgRight)

    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))

    # Plot the result
    return disparity

if __name__ == '__main__':
    #imgLeft = cv2.imread(r"C:\\Users\\guyju\\source\\repos\\DepthMap\\DepthMap\\CameraCalibTest\\imgL11.jpg")
    #imgRight = cv2.imread(r"C:\\Users\\guyju\\source\\repos\\DepthMap\\DepthMap\\CameraCalibTest\\imgR11.jpg")
    #imgR_gray = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
    #imgL_gray = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 2, 1)
    #plt.imshow(imgLeft)
    #plt.axis('off')
    #plt.subplot(1, 2, 2)
    #plt.imshow(imgRight)
    #plt.axis('off')
    #plt.show()
    #result = ShowDisparity(imgL_gray, imgR_gray, bSize=5)
    #plt.imshow(result)
    #plt.axis('off')
    #plt.show()

    #Calibrate
    calibration()

    disp_map = np.zeros((800, 800, 3))
    CamL_id = 0
    CamR_id = 1
    CamL = cv2.VideoCapture(CamL_id)
    CamR = cv2.VideoCapture(CamR_id)
    if CamL.isOpened():
        print("Camera with index : L is opened")
    if CamR.isOpened():
        print("Camera with index : R is opened")
    cv_file = cv2.FileStorage(".Deepmap\\dataparamspy.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    Baseline = cv_file.getNode("Trns").mat()
    cv_file.release()
    print("Baseline: ")
    print(Baseline)
    #numDisparities = int(16)
    #blockSize = int(5)
    #preFilterType = int(1)
    #preFilterSize = int(5)
    #preFilterCap = int(31)
    #textureThreshold = int(10)
    #uniquenessRatio = int(15)
    #speckleRange = int(0)
    #speckleWindowSize = int(0)
    #disp12MaxDiff = int(-1)
    #minDisparity = int(5)

    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 600, 600)
    cv2.namedWindow('left image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('left image', 800, 800)

    cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
    cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
    cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
    cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
    cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
    cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
    cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)

    print("Starting video ...")
    stereo = cv2.StereoBM_create()
    while True:
        retL, imgL = CamL.read()
        retR, imgR = CamR.read()

        if retL and retR:
            imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            Left_nice = cv2.remap(imgL_gray,
                                  Left_Stereo_Map_x,
                                  Left_Stereo_Map_y,
                                  cv2.INTER_LANCZOS4,
                                  cv2.BORDER_CONSTANT,
                                  0)

            # Applying stereo image rectification on the right image
            Right_nice = cv2.remap(imgR_gray,
                                   Right_Stereo_Map_x,
                                   Right_Stereo_Map_y,
                                   cv2.INTER_LANCZOS4,
                                   cv2.BORDER_CONSTANT,
                                   0)
            numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
            blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
            preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
            preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
            preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
            textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
            uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
            speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
            speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
            disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
            minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

            stereo.setNumDisparities(numDisparities)
            stereo.setBlockSize(blockSize)
            stereo.setPreFilterType(preFilterType)
            stereo.setPreFilterSize(preFilterSize)
            stereo.setPreFilterCap(preFilterCap)
            stereo.setTextureThreshold(textureThreshold)
            stereo.setUniquenessRatio(uniquenessRatio)
            stereo.setSpeckleRange(speckleRange)
            stereo.setSpeckleWindowSize(speckleWindowSize)
            stereo.setDisp12MaxDiff(disp12MaxDiff)
            stereo.setMinDisparity(minDisparity)


            disparity = stereo.compute(imgL_gray, imgR_gray)
            # Converting to float32
            #disparity = disparity.astype(np.float32)
            min = disparity.min()
            max = disparity.max()
            disparity = np.uint8(255 * (disparity - min) / (max - min))
            # Scaling down the disparity values and normalizing them
            #disparity = (disparity / 16.0 - minDisparity) / numDisparities
            cv2.imshow("disp", disparity)
            cv2.imshow("left image", imgL)

            if cv2.waitKey(1) == 27:
                break

