#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-28 下午8:22
# @Author  : 胡大为
# @Site    :
# @File    : build_records.py
# @Software: LukaObjectDetect

# [INFO] processing train...
# [INFO] 1363 examples saved for ’train’
# [INFO] processing test...
# [INFO] 241 examples saved for ’test’


# [INFO]
# [INFO] 15637 examples saved for ’train’
# [INFO] 2847 examples saved for ’test’
import pipeline.config as config
from pipeline.tfannotation import TFAnnotation
from bs4 import BeautifulStoneSoup
import tensorflow as tf
import os
from imutils import paths
import glob
import random
# 4282 examples saved for ’train’
# 778 examples saved for ’test’
# python /home/ices/.virtualenvs/dl4cv/lib/python3.5/site-packages/tensorflow/models/research/object_detection/model_main.py --pipeline_config_path=model/pipeline.config --model_dir=experiments/train --alsologtostderr


def main(_):
    f=open(config.CLASSES_FILE,'w')
    for (k,v) in config.CLASSES.items():
        item = ("item {\n"
            "\tid: " + str(v) + "\n"
            "\tname: '" + k + "'\n"
            "}\n")
        f.write(item)
    f.close()

    imagePaths = list(paths.list_images(config.IMAGE_PATH))
    random.shuffle(imagePaths)

    trainPaths = imagePaths[:int(len(imagePaths) * 0.85)]
    testPaths = imagePaths[int(len(imagePaths) * 0.85):]

    datasets = [
        ("train", trainPaths, config.TRAIN_RECORD),
        ("test", testPaths, config.TEST_RECORD)
    ]

    for (dType, inputPaths, outputPath) in datasets:
        print("[INFO] processing {}...".format(dType))
        writer = tf.python_io.TFRecordWriter(outputPath)
        total = 0
        file_count = 0
        for imagePath in inputPaths:
            try:
                xmlPath = imagePath[:imagePath.rfind(".")] + '.xml'
                if not os.path.exists(xmlPath):
                    continue
                contents = open(xmlPath, "r").read()
                soup = BeautifulStoneSoup(contents)
                file_count = file_count + 1
                encoded = tf.gfile.GFile(imagePath, "rb").read()
                encoded = bytes(encoded)

                (w, h, d) = (int(soup.find("width").contents[0]),
                             int(soup.find("height").contents[0]),
                             int(soup.find("depth").contents[0]))

                filename = soup.find("filename").contents[0]
                encoding = filename[filename.rfind(".") + 1:]

                tfAnnot = TFAnnotation()
                tfAnnot.image = encoded
                tfAnnot.encoding = encoding
                tfAnnot.filename = filename
                tfAnnot.width = w
                tfAnnot.height = h

                for object in soup.find_all("object"):
                    if len(object.find_all("bndbox")) > 1:
                        print("bndbox in 1 object > 1")
                    startX = max(0, float(object.find("xmin").contents[0]))
                    startY = max(0, float(object.find("ymin").contents[0]))
                    endX = min(w, float(object.find("xmax").contents[0]))
                    endY = min(h, float(object.find("ymax").contents[0]))
                    label = object.find("name").contents[0]
                    xMin = startX / w
                    xMax = endX / w
                    yMin = startY / h
                    yMax = endY / h
                    if xMin > xMax or yMin > yMax:
                        continue
                    tfAnnot.xMins.append(xMin)
                    tfAnnot.xMaxs.append(xMax)
                    tfAnnot.yMins.append(yMin)
                    tfAnnot.yMaxs.append(yMax)
                    tfAnnot.textLabels.append(label.encode("utf8"))
                    tfAnnot.classes.append(config.CLASSES[label])
                    tfAnnot.difficult.append(0)
                    total += 1
                features = tf.train.Features(feature=tfAnnot.build())
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
            except Exception as e:
                print(e)
                print(imagePath)
                continue

        writer.close()
        print("[INFO] {} examples saved for ’{}’".format(total, dType))
        print("[INFO] dealing with {} files for ’{}’".format(file_count, dType))

if __name__ == "__main__":
    tf.app.run()