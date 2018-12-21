#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-6 下午3:56
# @Author  : ManWingloeng
# @Site    : 
# @File    : testycrcb.py
# @Software: readingbook

import cv2
# from imutils.video.webcamvideostream import WebcamVideoStream
import numpy as np



frame = cv2.imread("/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/data/all_nail/test/05.jpg")
frame=cv2.blur(frame,(11,11))
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
print(frame.shape[:2])
# mask=np.zeros(shape=frame.shape[:2])
# for x in range(frame.shape[0]):
#     for y in range(frame.shape[1]):
#         phx=frame[x][y]
#         if (phx[2]>=100 and phx[2]<=120 and phx[1]>=140 and phx[1]<=175):
#             mask[x][y]=1
# mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,(11,11))
# mask=cv2.dilate(mask,(11,11))
# mask=cv2.erode(mask,(5,5))
# image=gray*mask

mask_frame=frame[:,:,1:]
mask=np.ones(shape=mask_frame.shape)
msk_lower=np.array([136, 100]) #140,100
msk_upper=np.array([175, 120]) #175,120
print("mask: ",mask.shape)
print("mask_frame: ",mask_frame.shape)
np.putmask(mask, mask_frame<msk_lower, 0.)
np.putmask(mask, mask_frame>msk_upper, 0.)

mask=mask[:,:,0]*mask[:,:,1]
# print(mask.shape)

mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,(11,11))
mask=cv2.dilate(mask,(11,11))
mask=cv2.erode(mask,(5,5))
image=gray*mask


cv2.namedWindow("img")
cv2.namedWindow("gray")
# cv2.namedWindow("BGR")
# cv2.namedWindow("Ycbcr")
cv2.imshow("img",frame)
cv2.imshow("gray",image)

# cv2.imshow("BGR",mask_frame_BGR)
# cv2.imshow("Ycbcr",mask_frame_gray)
if cv2.waitKey(1000) & 0xFF == ord("q"):
    cv2.destroyAllWindows()

