#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-10 上午10:12
# @Author  : 胡大为&文永亮
# @Site    :
# @File    : wrapper_for_predict_nail.py
# @Software: ImageNetBundle

# import the necessary packages
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import imutils
import cv2


def predict_nail(img):
    args={
        "model":"../model/export_model_008/frozen_inference_graph.pb",
        # "labels":"../record/classes.pbtxt",
        # "num_classes":1,
        "min_confidence":0.7}

    model=tf.Graph()

    with model.as_default():
        graphDef=tf.GraphDef()

        with tf.gfile.GFile(args["model"],"rb" ) as f:
            serializedGraph=f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef,name="")

    # labelMap=label_map_util.load_labelmap(args["labels"])
    # categories=label_map_util.convert_label_map_to_categories(
    #     labelMap,max_num_classes=args["num_classes"],use_display_name=True
    # )
    # categoryIdx=label_map_util.create_category_index(categories)

    with model.as_default():
        with tf.Session(graph=model) as sess:
            imageTensor=model.get_tensor_by_name("image_tensor:0")
            boxesTensor=model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score
            # (i.e., probability) and class label
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")

            predict = []
            frame = img
            if frame is None:
                return predict 
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

            # run the trained model to predict
            (boxes, scores, labels, N) = sess.run(  
                [boxesTensor, scoresTensor, classesTensor, numDetections],
                feed_dict={imageTensor: image})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)

            for (box, score, label) in zip(boxes, scores, labels):
                if score < args["min_confidence"]:
                    continue
                # scale the bounding box from the range [0, 1] to [W, H]
                (startY, startX, endY, endX) = box
                # compute the midX, midY for predict
                midX = ((endX + startX) * W)/2.
                midY = ((endY + startY) * H)/2.
                predict.append((midX, midY))
    # return the predict coordinate list of the img 
    return predict

### for test
test_img_path = "/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/data/all_nail/test/01.jpg"
test_img = cv2.imread(test_img_path)
print(predict_nail(test_img))



