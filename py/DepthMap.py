import numpy as np
import cv2 as cv


def nothing(x):
    pass

if __name__ == '__main__':
    disp_map = np.zeros((800, 800, 3))
    CamL_id = 1
    CamR_id = 2
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
    baseline = cv_file.getNode("Baseline").mat()[0][0]
    cv_file.release()
    print("Baseline: ")
    print(baseline)

    print("Starting video ...")
    stereo = cv.StereoSGBM.create()
    while True:
        retL, imgL = CamL.read()
        retR, imgR = CamR.read()

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

            hr, wr, dr = Right_nice.shape
            hl, wl, dl = Left_nice.shape

            if wr == wl:
                f_pixel = (wr * 0.5) / np.tan(70 * 0.5 * np.pi / 180)
            else:
                print('Left and right camera frames do not have the same pixel width')
            disparity = stereo.compute(imgL, imgR)
            disparity = cv.erode(disparity, None, iterations=1)
            disparity = cv.dilate(disparity, None, iterations=1)
            disparity_normal = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
            # Scaling down the disparity values and normalizing them
            # disparity = (disparity / 16.0 - minDisparity) / numDisparities


            # depth = np.float32((baseline * f_pixel) / disparity_normal)


            image = np.array(disparity_normal, dtype=np.uint8)
            disparity_color = cv.applyColorMap(image, cv.COLORMAP_BONE)

            # Show depth map
            cv.imshow("Depth map", np.hstack((disparity_color, imgL)))

            cv.imshow("disp", disparity)
            if cv.waitKey(1) == 27:
                break

    # Release and destroy all windows before termination
    CamL.release()
    CamR.release()
    cv.destroyAllWindows()
