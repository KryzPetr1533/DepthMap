import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap2.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

num = 1

while cap.isOpened():
    succes1, imgL = cap.read()
    succes2, imgR = cap2.read()

    k = cv2.waitKey(1)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('./CameraCalibTest/imgL' + str(num) + '.jpg', imgR)
        cv2.imwrite('./CameraCalibTest/imgR' + str(num) + '.jpg', imgL)
        print("images saved!")
        num += 1
    cv2.imshow('Right Image', imgR)
    cv2.imshow('Left Image', imgL)
    stacked = np.hstack([cv2.resize(imgL, (int(imgL.shape[0] / 2), int(imgL.shape[1] / 2))),
                         cv2.resize(imgR, (int(imgL.shape[0] / 2), int(imgL.shape[1] / 2)))])
    #stacked = cv2.resize(stacked, (int(imgL.shape[0] / 2), int(imgL.shape[1] / 2)))
    cv2.imshow('Stacked', stacked)



# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()