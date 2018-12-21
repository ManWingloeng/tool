#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-27 下午3:18
# @Author  : 胡大为
# @Site    : 
# @File    : config.py
# @Software: readingbook
import os

# BASE_PATH="/media/ices/0005A94600082E56/PycharmProjects/luka/readingbook"
BASE_PATH="/home/ices/yl_tmp/readingbook"
CLASSES_FILE=os.path.sep.join([BASE_PATH,"record","classes.pbtxt"])
CLASSES={"nail":1}
IMAGE_PATH=os.path.sep.join([BASE_PATH,"data","all_nail","nail_image"])
TRAIN_RECORD=os.path.sep.join([BASE_PATH,"record","train_all.rec"])
TEST_RECORD=os.path.sep.join([BASE_PATH,"record","test_all.rec"])
