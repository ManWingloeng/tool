# coding:utf-8
#PRESS Q to shut down all windows displayed on running the program.

import numpy as np
import cv2

# cap=cv2.VideoCapture(0)
# while(True):
# .read() returns two values, the second one is the image
# ret, frame = cap.read()
frame=cv2.imread("/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/data/all_nail/test/05.jpg")
frame=np.asarray(frame)

#============finding the area of the frame that has skin color============================

lower = np.array([0, 133, 77], dtype="uint8")
upper = np.array([255, 177, 127], dtype="uint8")
frmskin = frame.copy()
frmskin = cv2.GaussianBlur(frmskin, (11, 11), 0)
frmskin = cv2.bilateralFilter(frmskin, 9, 75, 100)
imageYCrCb = cv2.cvtColor(frmskin, cv2.COLOR_BGR2YCR_CB)
skinmask = cv2.inRange(imageYCrCb, lower, upper)
ret,bin_mask=cv2.threshold(skinmask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_,csk,_=cv2.findContours(bin_mask.copy(),1,2)
skin_cmax = max(csk, key=cv2.contourArea)
skin_area=cv2.contourArea(skin_cmax)

#===================Applying threshold and smoothening the image===============================

frameblur = cv2.GaussianBlur(frame.copy(), (21, 21), 0)
gray = cv2.cvtColor(frameblur, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
thresh = cv2.bilateralFilter(thresh, 9, 75, 100)

#==================noise removal and applying skin mask=========================================

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1, iterations=2)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations=2)
closing = cv2.dilate(closing, kernel2, iterations=2)
skin_closed= cv2.bitwise_and(closing,bin_mask)

#===========finding contours and ellipse=====================================================

_, contours, _= cv2.findContours(skin_closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
fist=skin_closed.copy()
cmax = max(contours, key=cv2.contourArea)

epsilon = 0.000001*cv2.arcLength(cmax,True)
cmax = cv2.approxPolyDP(cmax,epsilon,True)
cv2.drawContours(frame,[cmax],0,(255,255,51),3)              #skyblue
cv2.imshow("cmax",frame)
if len(cmax) >= 5:                                           #if points in cmax are less than 5 the fitellipse throws error
    ellipse = cv2.fitEllipse(cmax)
    pc, lt, angle = ellipse
ratio_axes=lt[0] / lt[1]
if ratio_axes > 0.4:
    mas_elli = ((pc[0],pc[1]), (1.3*lt[0], 0.75*lt[1]), angle)
else:
    mas_elli = ((pc[0], pc[1]), (1.4 * lt[0], 0.85 * lt[1]), angle)
fist_frm = frame.copy()
cv2.ellipse(frame, mas_elli, (0, 0, 255), 5)

#=================== checking if fingers are open ================================================

kernelf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
fist = cv2.morphologyEx(fist, cv2.MORPH_OPEN, kernelf, iterations=3)
fist = cv2.dilate(fist, kernelf, iterations=2)
fist = cv2.erode(fist, kernelf, iterations=4)
_, contoursf, _ = cv2.findContours(fist, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if len(contoursf) >= 1:
    cmax_fist=max(contoursf, key=cv2.contourArea)
    if len(cmax_fist) >= 5:
        ellipse_fist = cv2.fitEllipse(cmax_fist)
        pc_fi, lt_fi, angle_fi = ellipse_fist
    elli_fist = ((pc_fi[0], pc_fi[1]), (3* lt_fi[0], 1.1 * lt_fi[1]), angle_fi)
    cv2.ellipse(fist_frm, elli_fist, (0, 0, 255), 5)
    blank_fist = np.zeros(fist_frm.shape, dtype=np.uint8)
    frm1_fist = cv2.drawContours(blank_fist.copy(), [cmax], 0, (255, 255, 255), 2)
    frm2_fist = cv2.ellipse(blank_fist.copy(), elli_fist, (255, 255, 255), 2)
    intersect_fist = cv2.bitwise_and(frm1_fist, frm2_fist)
    intersect_fist = cv2.cvtColor(intersect_fist, cv2.COLOR_RGB2GRAY)
    _, contoursp_fist, _ = cv2.findContours(intersect_fist, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    pts_fist = (len(contoursp_fist))
    pts_fist=pts_fist/2

#===============finding pts of intersection=====================================================
blank = np.zeros(frame.shape,dtype=np.uint8)
frm1 = cv2.drawContours(blank.copy(), [cmax], 0, (255,255,255),2)
frm2 = cv2.ellipse(blank.copy(), mas_elli, (255,255,255),2)
intersect = cv2.bitwise_and(frm1, frm2)
intersect=cv2.cvtColor(intersect,cv2.COLOR_RGB2GRAY)
_, contoursp, _ = cv2.findContours(intersect, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
pts=(len(contoursp))
fingers=int((pts/2)-1)

#===============displaying results==============================================================
if skin_area<20000:
    cv2.putText(frame, str(-1), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
elif (pts_fist < 2) :
    cv2.putText(frame, str(0), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
else:
    cv2.putText(frame, str(fingers), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
#cv2.imshow('intersect',intersect)
cv2.imshow('frame',np.hstack([frame]))
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    # break

# cap.release()
# cv2.destroyAllWindows()