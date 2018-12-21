#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-6 下午3:56
# @Author  : 胡大为
# @Site    : 
# @File    : test.py
# @Software: readingbook

import cv2
from imutils.video.webcamvideostream import WebcamVideoStream
import numpy as np

web=WebcamVideoStream(src=0)
web.start()

while True:
    frame=web.read()
    fram=cv2.blur(frame,(11,11))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    print(frame.shape)
    mask=np.zeros(shape=frame.shape[:2])
    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            phx=frame[x][y]
            if (phx[2]>=100 and phx[2]<=120 and phx[1]>=140 and phx[1]<=175):
                mask[x][y]=1

    # np.putmask(frame, )
    # np.copyto
    # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,(11,11))
    mask=cv2.dilate(mask,(11,11))
    mask=cv2.erode(mask,(5,5))
    image=gray*mask
    cv2.imshow("ori",gray)
    cv2.imshow("face",image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
web.stop()
