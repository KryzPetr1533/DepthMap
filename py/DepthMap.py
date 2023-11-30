import numpy as np
import cv2 as cv


def nothing(x):
    pass


if __name__ == '__main__':
    CamL_id = 0
    CamR_id = 1
    CamL = cv.VideoCapture(CamL_id)
    CamL.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    CamL.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    CamR = cv.VideoCapture(CamR_id)
    CamR.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    CamR.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    if CamL.isOpened():
        print("Camera with index : L is opened")
    if CamR.isOpened():
        print("Camera with index : R is opened")
    cv_file = cv.FileStorage(".\\dataparamspy.xml", cv.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    baseline = cv_file.getNode("Baseline").mat()[0][0]
    left_cam_mat = cv_file.getNode("Left_cam_mat").mat()
    right_cam_mat = cv_file.getNode("Right_cam_mat").mat()
    cv_file.release()
    print("Baseline: ")
    print(baseline)
    print("Left camera matrix:")
    print(left_cam_mat)
    print("Right camera matrix:")
    print(right_cam_mat)
    cv.namedWindow('trackbar', cv.WINDOW_NORMAL)
    cv.createTrackbar('preFilterCap', 'trackbar', 10, 70, nothing)
    cv.createTrackbar('SADWindowSize', 'trackbar', 1, 10, nothing)
    cv.createTrackbar('minDisparity', 'trackbar', 0, 20, nothing)
    cv.createTrackbar('numberOfDisparities', 'trackbar', 1, 10, nothing)
    cv.createTrackbar('P', 'trackbar', 1, 100, nothing)
    cv.createTrackbar('uniquenessRatio', 'trackbar', 5, 100, nothing)
    cv.createTrackbar('speckleRange', 'trackbar', 1, 100, nothing)
    cv.createTrackbar('speckleWindowSize', 'trackbar', 25, 200, nothing)
    cv.createTrackbar('disp12MaxDiff', 'trackbar', 1, 25, nothing)
    stereo = cv.StereoSGBM.create()
    # while cv.waitKey(5) != 27:
    # hr, wr, dr = imgR.shape
    # hl, wl, dl = imgL.shape
    #
    # if wr == wl:
    #     f_pixel = (wr * 0.5) / np.tan(70 * 0.5 * np.pi / 180)
    # else:
    #     print('Left and right camera frames do not have the same pixel width')

    while cv.waitKey(1) != 27:
        retL, imgL = CamL.read()
        retR, imgR = CamR.read()

        preFilterCap = cv.getTrackbarPos('preFilterCap', 'trackbar')
        SADWindowSize = cv.getTrackbarPos('SADWindowSize', 'trackbar') * 2 + 1
        minDisparity = cv.getTrackbarPos('minDisparity', 'trackbar')
        numberOfDisparities = cv.getTrackbarPos('numberOfDisparities', 'trackbar') * 16
        P1 = cv.getTrackbarPos('P', 'trackbar') * 8 * SADWindowSize * SADWindowSize
        P2 = cv.getTrackbarPos('P', 'trackbar') * 32 * SADWindowSize * SADWindowSize
        uniquenessRatio = cv.getTrackbarPos('uniquenessRatio', 'trackbar')
        speckleRange = cv.getTrackbarPos('speckleRange', 'trackbar')
        speckleWindowSize = cv.getTrackbarPos('speckleWindowSize', 'trackbar')
        disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff', 'trackbar')

        stereo.setPreFilterCap(preFilterCap)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        if retL and retR:
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

            # left_grey = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
            # right_grey = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
            # imgL = cv.resize(imgL, (wr, hr))
            # imgR = cv.resize(imgL, (wr, hr))
            # Left_nice = cv.erode(Left_nice, None, iterations=1)
            # Left_nice = cv.dilate(Left_nice, None, iterations=1)
            # Left_nice = cv.blur(Left_nice, (5, 5))
            # Right_nice = cv.blur(Right_nice, (5, 5))
            disparity = stereo.compute(imgL, imgR).astype(np.float32)/16/numberOfDisparities
            # print(np.shape(disparity))
            # disparity = cv.blur(disparity, (5, 5))
            # disparity = cv.erode(disparity, None, iterations=1)
            # disparity = cv.dilate(disparity, None, iterations=1)
            disparity_normal = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)

            hr, wr, dr = Left_nice.shape
            # hl, wl, dl = imgL.shape
            f_pixel = left_cam_mat[0][0]
            #print(len(disparity_normal), len(disparity_normal[0]))
            disparity[disparity == 0] = 1
            depth = (baseline * f_pixel) / disparity
            depth_normal = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX)
            image = np.array(disparity, dtype=np.uint8)
            # depth_image = np.array(depth_normal, dtype=np.uint8)
            disparity_color = cv.applyColorMap(image, cv.COLORMAP_JET)
            # depth_color = cv.applyColorMap(depth_normal, cv.COLORMAP_JET)
            print('')
            # Show depth map
            cv.imshow("Depth", depth)
            cv.imshow("Disparity", disparity_color)
            cv.imshow("images", imgL)

    #Save data into txt file
    np.set_printoptions(threshold=np.inf)
    file = open('depth.txt', 'w')
    file.write(np.array2string(depth, max_line_width=None))
    file.close()
    file = open('disparity.txt', 'w')
    file.write(np.array2string(disparity, max_line_width=None))
    file.close()
    # file = open('disparity_normal.txt', 'w')
    # file.write(np.array2string(disparity_normal, max_line_width=None))
    #file.close()
    np.save('depth', depth)
    cv.imwrite('test.jpg', Left_nice)

# Release and destroy all windows before termination
CamL.release()
CamR.release()
cv.destroyAllWindows()
