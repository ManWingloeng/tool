#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-10 上午10:12
# @Author  : 胡大为,ManWingloeng
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

with model.as_default():
    with tf.Session(graph=model) as sess:
        imageTensor=model.get_tensor_by_name("image_tensor:0")
        boxesTensor=model.get_tensor_by_name("detection_boxes:0")

        # for each bounding box we would like to know the score
        # (i.e., probability) and class label
        scoresTensor = model.get_tensor_by_name("detection_scores:0")
        classesTensor = model.get_tensor_by_name("detection_classes:0")
        numDetections = model.get_tensor_by_name("num_detections:0")

        # vs=WebcamVideoStream(src=0)
        # vs.start()
        # while True:
        imgs_test_path="/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/data/all_nail/test"
        import os
        file_names = os.listdir(imgs_test_path)
        imgs = []
        for img_name in file_names:
            img_path = os.path.join(imgs_test_path, img_name)
            imgs.append(img_path)


        for img in imgs: 
            frame = cv2.imread(img)
            #457.jpg
            #(box: [0.3738367  0.6832962  0.41337186 0.72378075], score: 0.492618203163147, label: 1.0)
            #(box: [0.5201001  0.75700814 0.5695447  0.793104  ], score: 0.4269217550754547, label: 1.0)

            image = frame
            # cv2.namedWindow("image")
            # cv2.imshow("image", image)
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


            print("test__________________")
            for (box, score, label) in zip(boxes, scores, labels):
                print("(box: {}, score: {}, label: {})".format(box, score, label))
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
                midX = (startX+endX)/2.
                midY= (startY+endY)/2.
                print("midX:",midX)
                print("midY:",midY)
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
            cv2.namedWindow("Output")
            cv2.imshow("Output", output)
            # cv2.waitKey(0)
            if cv2.waitKey(1)&0xFF==ord("q"):
                break


        cv2.destroyAllWindows()
        # vs.stop()
