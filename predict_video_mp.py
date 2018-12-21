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
import pipeline.config as config
import random
from imutils.video import WebcamVideoStream
import multiprocessing as mp

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
    "min_confidence":0.6}

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


# url = "http://admin:admin@10.85.2.241:8081/"
# cap = cv2.VideoCapture(url)










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
        cap = cv2.VideoCapture(url)

        def queue_img_update(q):
            while True:
                is_opened, frame = cap.read()
                if is_opened:
                    q.put(frame)
                if q.qsize() > 1:
                    q.get()
        # vs=WebcamVideoStream(src=0)
        # vs.start()
        def queue_img_get(q):
            while True:
                frame=q.get()
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
                image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)

                (boxes, scores, labels, N) = sess.run(
                    [boxesTensor, scoresTensor, classesTensor, numDetections],
                    feed_dict={imageTensor: image})
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                labels = np.squeeze(labels)

                for (box, score, label) in zip(boxes, scores, labels):
                    # print(int(label))
                    # if int(label) != 1:
                    #     continue
                    if score < args["min_confidence"]:
                        continue
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
                cv2.waitKey(1)
                # if cv2.waitKey(1)&0xFF==ord("q"):
                #     break
            # cv2.destroyAllWindows()


            mp.set_start_method(method='spawn')
            queue = mp.Queue(maxsize=2)
            queue_img_update(queue)
            queue_img_get(queue)
            processes = [mp.Process(target=queue_img_update, args=(queue)), mp.Process(target=queue_img_get, args=(queue))]
            [setattr(process, "daemon", True) for process in processes]
            [process.start() for process in processes]
            [process.join() for process in processes]
        # vs.stop()
