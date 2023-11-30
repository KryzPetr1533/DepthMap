import numpy as np
import cv2 as cv

depth = np.load('depth.npy')
np.set_printoptions(threshold=np.inf)
aim = depth[np.logical_and(depth < 200000000, depth > 231219)]
pos = np.where(depth == aim[0])
print(pos)
print(type(depth))
mask = depth.copy()
mask[np.logical_or(depth > 200000000, depth < 231219)] = 0
photo = cv.imread('test.jpg')
print(photo.shape, mask.shape)
gphoto = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
while cv.waitKey(1) != 27:
    cv.imshow('depth', depth)
    cv.imshow('mask', mask)
    cv.imshow('photo', photo)