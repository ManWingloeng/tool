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


class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(URL)

    def start(self):
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        self.isstop = True
        print('ipcam stopped!')

    def getframe(self):
        return self.Frame

    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()

        self.capture.release()


# url = "http://admin:admin@10.85.12.79:8081/"
# url = 0
# ipcam = ipcamCapture(url)
# ipcam.start()
# time.sleep(1)
# cap = cv2.VideoCapture(0)
# vs=WebcamVideoStream(src=0)
# vs.start()
while_cnt = 0
# vibe = Vibe()



web = WebcamVideoStream(src=0)
web.start()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
# fgbg_old = cv2.bgsegm.createBackgroundSubtractorGMG()
# fgbg.setDecisionThreshold(40)
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
if web.stream.isOpened():
    while True:
        time_s = time.time()
        while_cnt += 1
        print("runing in :", while_cnt, "step")
        # frame = ipcam.getframe()
        # rett, frame = cap.read()
        frame = web.read()
        # cam = cv2.VideoCapture(url)
        # status, frame = cam.read()
        if frame is None:
            continue
        image = frame.copy()
        print(len(image))
        print(image[0].shape)
        N = 20
        R = 20
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

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray",gray)
        # gray = np.expand_dims(gray, axis=2)
        # gray = gray.astype(np.int32)

        skin = fdfg.test(image)

        img, bin_mask, res_hand, cmax = fdfg.find_hand(image)
        img_old, bin_mask_old, res_hand_old = fdfg.find_hand_old(image)
        time_find_hand = time.time() - time_s
        print("time_find_hand:", time_find_hand)
        # gray_res_hand = cv2.cvtColor(res_hand, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray_res_hand", gray_res_hand)
        # gray_res_hand = np.expand_dims(gray_res_hand, axis=2)
        # gray_res_hand = gray_res_hand.astype(np.int32)

        ### MOG2 ###################################
        cv2.imshow("res_hand", res_hand)
        # cv2.imshow("res_hand_old", res_hand_old)
        # fgmask_old = fdfg.
        # fgmask = fgbg.apply(res_hand_old)
        # fgmask_old = fgbg_old.apply(res_hand_old)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # fgmask_old = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("GMG", fgmask)
        # cv2.imshow("GMG_old", fgmask_old)
        fgmask2 = fgbg2.apply(res_hand)

        cv2.imshow("MOG2", fgmask2)
        #### vibe+ ###############################
        res_hand = res_hand.astype(np.int32)
        if while_cnt == 1:
            samples = c_func.func1(res_hand, N, window)
        # # segMap, samples = fv.vibe_detection(gray, samples, _min, N, R)
        move_obj, samples = c_func.func2(res_hand, samples, _min, N, R, phai, window)
        time_chafen = time.time() - time_s
        print("time_chafen:", time_chafen)
        print("move_obj.shape", move_obj.shape)
        move_obj = np.array(move_obj, dtype=np.uint8)
        move_obj = cv2.morphologyEx(move_obj, cv2.MORPH_OPEN, kernel)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # move_obj = cv2.erode(move_obj, kernel, iterations=3)
        # move_obj = cv2.dilate(move_obj, kernel, iterations=2)
        # move_obj_closed = cv2.morphologyEx(move_obj, cv2.MORPH_OPEN, kernel)
        # print("move_obj", move_obj_closed)
        # print("samples.shape", samples.shape)
        # print("samples", samples)
        cv2.namedWindow("move_obj")
        cv2.imshow("move_obj", move_obj)
        ##########################################
        # and_move_bin = cv2.bitwise_and(move_obj, bin_mask)
        # cv2.imshow("and_move_bin",and_move_bin)

        fdfg.gravity(img, bin_mask, cmax)
        # cv2.namedWindow("bin_mask")
        # cv2.imshow("bin_mask", bin_mask)
        cv2.namedWindow("img")
        cv2.imshow("img", img)
        # show the output image
        cv2.imshow("Output", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            # ipcam.stop()
            break
