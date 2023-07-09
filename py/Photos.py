import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

num = 1

while cap.isOpened():
    succes1, imgL = cap.read()
    succes2, imgR = cap2.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('./CameraCalibTest/imgL' + str(num) + '.jpg', imgR)
        cv2.imwrite('./CameraCalibTest/imgR' + str(num) + '.jpg', imgL)
        print("images saved!")
        num += 1
    cv2.imshow('Right Image', imgR)
    cv2.imshow('Left Image', imgL)

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()