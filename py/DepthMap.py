import numpy as np
import cv2 as cv


def nothing(x):
    pass


if __name__ == '__main__':
    disp_map = np.zeros((800, 800, 3))
    CamL_id = 0
    CamR_id = 1
    CamL = cv.VideoCapture(CamL_id)
    CamR = cv.VideoCapture(CamR_id)
    if CamL.isOpened():
        print("Camera with index : L is opened")
    if CamR.isOpened():
        print("Camera with index : R is opened")
    cv_file = cv.FileStorage(".\\dataparamspy.xml", cv.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    Baseline = cv_file.getNode("Trns").mat()
    cv_file.release()
    print("Baseline: ")
    print(Baseline)

    # cv.namedWindow('disp', cv.WINDOW_NORMAL)
    # cv.resizeWindow('disp', 600, 600)
    # cv.namedWindow('left image', cv.WINDOW_NORMAL)
    # cv.resizeWindow('left image', 800, 800)
    #
    # cv.createTrackbar('numDisparities', 'disp', 2, 17, nothing)
    # cv.createTrackbar('blockSize', 'disp', 5, 50, nothing)
    # cv.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
    # cv.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
    # cv.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
    # cv.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
    # cv.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
    # cv.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
    # cv.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
    # cv.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
    # cv.createTrackbar('minDisparity', 'disp', 5, 25, nothing)

    print("Starting video ...")
    stereo = cv.StereoSGBM.create(mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
    while True:
        retL, imgL = CamL.read()
        retR, imgR = CamR.read()

        if retL and retR:
            # imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
            # imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
            # Applying stereo image rectification on the left image
            Left_nice = cv.remap(imgL,
                                 Left_Stereo_Map_x,
                                 Left_Stereo_Map_y,
                                 cv.INTER_LANCZOS4,
                                 cv.BORDER_CONSTANT,
                                 0)

            # Applying stereo image rectification on the right image
            Right_nice = cv.remap(imgR,
                                  Right_Stereo_Map_x,
                                  Right_Stereo_Map_y,
                                  cv.INTER_LANCZOS4,
                                  cv.BORDER_CONSTANT,
                                  0)

            # numDisparities = cv.getTrackbarPos('numDisparities', 'disp') * 16
            # blockSize = cv.getTrackbarPos('blockSize', 'disp') * 2 + 5
            # preFilterSize = cv.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
            # preFilterCap = cv.getTrackbarPos('preFilterCap', 'disp')
            # textureThreshold = cv.getTrackbarPos('textureThreshold', 'disp')
            # uniquenessRatio = cv.getTrackbarPos('uniquenessRatio', 'disp')
            # speckleRange = cv.getTrackbarPos('speckleRange', 'disp')
            # speckleWindowSize = cv.getTrackbarPos('speckleWindowSize', 'disp') * 2
            # disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff', 'disp')
            # minDisparity = cv.getTrackbarPos('minDisparity', 'disp')
            #
            # stereo.setNumDisparities(numDisparities)
            # stereo.setBlockSize(blockSize)
            # stereo.setPreFilterSize(preFilterSize)
            # stereo.setPreFilterCap(preFilterCap)
            # stereo.setTextureThreshold(textureThreshold)
            # stereo.setUniquenessRatio(uniquenessRatio)
            # stereo.setSpeckleRange(speckleRange)
            # stereo.setSpeckleWindowSize(speckleWindowSize)
            # stereo.setDisp12MaxDiff(disp12MaxDiff)
            # stereo.setMinDisparity(minDisparity)

            hr, wr, dr = Right_nice.shape
            hl, wl, dl = Left_nice.shape

            if wr == wl:
                f_pixel = (wr * 0.5) / np.tan(70 * 0.5 * np.pi / 180)
            else:
                print('Left and right camera frames do not have the same pixel width')
            disparity = stereo.compute(imgL, imgR)

            # Converting to float32
            disparity = disparity.astype(np.float32)

            # Scaling down the disparity values and normalizing them
            # disparity = (disparity / 16.0 - minDisparity) / numDisparities

            depth = np.float32((11 * f_pixel) / disparity)
            cv.imshow("disp", disparity)
            cv.imshow("depth", depth)
            cv.imshow("left image", imgL)
            cv.imshow("Left_nice", Left_nice)
            if cv.waitKey(1) == 27:
                break

    # Release and destroy all windows before termination
    CamL.release()
    CamR.release()
    cv.destroyAllWindows()
