#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-10 上午10:12
# @Author  : 胡大为
# @Site    :
# @File    : predict.py
# @Software: ImageNetBundle

# import the necessary packages
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
from imutils import paths
# import pipeline.config as config
import random
from imutils.video import WebcamVideoStream
import time
import threading
import time
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# help="base path for frozen checkpoint detection graph")
# ap.add_argument("-l", "--labels", required=True,
# help="labels file")
# ap.add_argument("-i", "--image", required=True,
# help="path to input image")
# ap.add_argument("-n", "--num-classes", type=int, required=True,
# help="# of class labels")
# ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
# help="minimum probability used to filter weak detections")
# args = vars(ap.parse_args())

# args={
#     "model":"../experiments/exported_model/frozen_inference_graph.pb",
#     # "model":"experiment/exported_model2/frozen_inference_graph.pb",
#       "labels":"../record/classes.pbtxt",
#     # "labels":"records/classes.pbtxt" ,
#       "num_classes":1,
#       "min_confidence":0.6}

args={
    # "model":"../experiments/exported_model/frozen_inference_graph.pb",
    "model":"/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/model/export_model_008/frozen_inference_graph.pb",
    # "model":"/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/model/export_model_015/frozen_inference_graph.pb",
    "labels":"../record/classes.pbtxt",
    # "labels":"record/classes.pbtxt" ,
    "num_classes":1,
    "min_confidence":0.7}

COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

model=tf.Graph()

with model.as_default():
    graphDef=tf.GraphDef()

    with tf.gfile.GFile(args["model"],"rb" ) as f:
        serializedGraph=f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef,name="")

labelMap=label_map_util.load_labelmap(args["labels"])
categories=label_map_util.convert_label_map_to_categories(
    labelMap,max_num_classes=args["num_classes"],use_display_name=True
)
categoryIdx=label_map_util.create_category_index(categories)


# import threading

# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
		
	# 攝影機連接。
        self.capture = cv2.VideoCapture(URL)

    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        
        self.capture.release()



with model.as_default():
    with tf.Session(graph=model) as sess:
        imageTensor=model.get_tensor_by_name("image_tensor:0")
        boxesTensor=model.get_tensor_by_name("detection_boxes:0")

        # for each bounding box we would like to know the score
        # (i.e., probability) and class label
        scoresTensor = model.get_tensor_by_name("detection_scores:0")
        classesTensor = model.get_tensor_by_name("detection_classes:0")
        numDetections = model.get_tensor_by_name("num_detections:0")

        url = "http://admin:admin@10.85.2.241:8081/"
        ipcam = ipcamCapture(url)
        ipcam.start()
        time.sleep(1)
        # vs=WebcamVideoStream(src=0)
        # vs.start()
        while True:
            frame = ipcam.getframe()
            if frame is None:
                continue
            image = frame
            (H, W) = image.shape[:2]
            # check to see if we should resize along the width
            if W > H and W > 1000:
                image = imutils.resize(image, width=1000)
            # otherwise, check to see if we should resize along the
            # height
            elif H > W and H > 1000:
                image = imutils.resize(image, height=1000)
            # prepare the image for detection
            (H, W) = image.shape[:2]
            output = image.copy()


            frame=cv2.blur(frame,(11,11))
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            # print(frame.shape[:2])
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
            msk_lower=np.array([140, 100]) #140,100
            msk_upper=np.array([175, 120]) #175,120
            # print("mask: ",mask.shape)
            # print("mask_frame: ",mask_frame.shape)
            np.putmask(mask, mask_frame<msk_lower, 0.)
            np.putmask(mask, mask_frame>msk_upper, 0.)

            mask=mask[:,:,0]*mask[:,:,1]
            # print(mask.shape)

            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,(11,11))
            mask=cv2.dilate(mask,(11,11))
            mask=cv2.erode(mask,(5,5))
            gray_image=gray*mask


            # cv2.namedWindow("img")
            cv2.namedWindow("gray")
            # cv2.namedWindow("BGR")
            # cv2.namedWindow("Ycbcr")
            # cv2.imshow("img",frame)
            cv2.imshow("gray",gray_image)



            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
            start=time.time()
            (boxes, scores, labels, N) = sess.run(
                [boxesTensor, scoresTensor, classesTensor, numDetections],
                feed_dict={imageTensor: image})
            print(time.time()-start)
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)

            for (box, score, label) in zip(boxes, scores, labels):
                # print(int(label))
                # if int(label) != 1:
                #     continue
                if score < args["min_confidence"]:
                    continue
                print("box: {}, score: {}, label: {}".format(box,score,label))
                # scale the bounding box from the range [0, 1] to [W, H]
                (startY, startX, endY, endX) = box
                startX = int(startX * W)
                startY = int(startY * H)
                endX = int(endX * W)
                endY = int(endY * H)
                # draw the prediction on the output image
                label = categoryIdx[label]
                idx = int(label["id"]) - 1
                label = "{}: {:.2f}".format(label["name"], score)
                cv2.rectangle(output, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(output, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)
            # show the output image
            cv2.imshow("Output", output)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                ipcam.stop()
                break

        # vs.stop()
