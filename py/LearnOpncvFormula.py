import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

def nothing(x):
    pass

if __name__ == '__main__':
    disp_map = np.zeros((800, 800, 3))
    CamL_id = 0
    CamR_id = 1
    CamL = cv2.VideoCapture(CamL_id)
    CamR = cv2.VideoCapture(CamR_id)
    if CamL.isOpened():
        print("Camera with index : L is opened")
    if CamR.isOpened():
        print("Camera with index : R is opened")
    cv_file = cv2.FileStorage(".\\dataparamspy.xml", cv2.FILE_STORAGE_READ)
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

            disparity = (disparity / 16.0 - minDisparity) / numDisparities
            cv2.imshow("disp", disparity)
            cv2.imshow("left image", imgL)

            if cv2.waitKey(1) == 27:
                break