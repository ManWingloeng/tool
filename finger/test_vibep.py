# -*- coding: utf-8 -*-
import numpy as np
import argparse
import imutils
import cv2
from imutils import paths
from imutils.video import WebcamVideoStream
import threading
import time
import find_finger as fdfg
# import finger.vibe as fv
# from finger.Vibe import Vibe
import c_func
# from c_vibep import ViBePlus
from ctypes import *
import numpy.ctypeslib as npct





while_cnt = 0

c_lib = cdll.LoadLibrary("./vibeplus/c/Cprintlibvibeplus.so")
vp = c_lib.Cvibep_new()

web = WebcamVideoStream(src=0)
web.start()
# vibeplus = ViBePlus()
while True:
    time_s = time.time()
    while_cnt += 1
    print("runing in :", while_cnt, "step")
    # frame = ipcam.getframe()
    # rett, frame = cap.read()
    frame = web.read()
    if frame is None:
        continue
    image = frame.copy()
    print(len(image))
    print(image[0].shape)
    N = 25
    R = 10
    _min = 2
    phai = 1
    window = 3
    # image=np.array(image)
    (H, W) = image.shape[:2]
    # check to see if we should resize along the width
    if W > H and W > 1000:
        image = imutils.resize(image, width=300)
    # otherwise, check to see if we should resize along the
    # height
    elif H > W and H > 1000:
        image = imutils.resize(image, height=300)
    # prepare the image for detection
    (H, W) = image.shape[:2]
    output = image.copy()
    img, bin_mask, res_hand, cmax = fdfg.find_hand(image)
    time_find_hand = time.time() - time_s
    print("time_find_hand:", time_find_hand)
    gray_res_hand = cv2.cvtColor(res_hand, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_res_hand", gray_res_hand)
#     cv2.imshow("gray_res_hand", gray_res_hand)
    # gray_res_hand = np.expand_dims(gray_res_hand, axis=2)
#     gray_res_hand = gray_res_hand.astype(np.int32)

    print(gray_res_hand.shape[0])
    print("ok here1")
    c_img_p = gray_res_hand.ctypes.data_as(POINTER(c_ubyte))
    print("ok here2")
    print(c_img_p)
    c_lib.CFrameCapture(vp, gray_res_hand.shape[0], gray_res_hand.shape[1], c_img_p)
#     c_lib.CFrameCapture(vp, gray_res_hand.shape[0], gray_res_hand.shape[1])
    print("ok here3")
    if(while_cnt == 1):
        c_lib.Cinit(vp)
#         c_lib.CProcessFirstFrame(vp)
#     CExtractBG(vp)
#     CCalcuUpdateModel(vp)
#     CUpdate(vp)
#     c_lib.CRun(vp)
#     SegModel = c_lib.CgetSegModel(vp)
#     UpdateModel = c_lib.CgetUpdateModel(vp)
    time_chafen = time.time() - time_s
    print("time_chafen:", time_chafen)

#     cv2.imshow("SegModel", SegModel)
#     cv2.imshow("UpdateModel", UpdateModel)
    fdfg.gravity(img, bin_mask, cmax)
#     cv2.namedWindow("bin_mask")
#     cv2.imshow("bin_mask", bin_mask)
    cv2.namedWindow("img")
    cv2.imshow("img", img)
    # show the output image
    cv2.imshow("Output", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break





