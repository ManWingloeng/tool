# coding:utf-8
import os
import cv2
import imghdr
import time
import os
import numpy as np
 
# preprocess YCrCb to covert the color
def pic_YCrCb(filepath, filePath_bak):
    files = []
    pathDir = os.listdir(filepath)
    print(pathDir)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        files.append(child)
    for index, file in enumerate(files):
        if imghdr.what(file) in ('jpg', 'png', 'jpeg'):
            print('preprocess img:%s'%file)
            frame = cv2.imread(file) #读取图片
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
            msk_lower=np.array([140, 100])
            msk_upper=np.array([175, 120])
            print("mask: ",mask.shape)
            print("mask_frame: ",mask_frame.shape)
            np.putmask(mask, mask_frame<msk_lower, 0.)
            np.putmask(mask, mask_frame>msk_upper, 0.)

            mask=mask[:,:,0]*mask[:,:,1]
            # print(mask.shape)

            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,(11,11))
            mask=cv2.dilate(mask,(11,11))
            mask=cv2.erode(mask,(5,5))
            image=gray*mask
            saving_path = os.path.join(filePath_bak, pathDir[index])
            print('saving in %s'%saving_path)
            
            cv2.imwrite(saving_path, image) #写入目录

 
if __name__ == '__main__':
    filePath = r"/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/data/all_nail/nail_image"
    filePath_bak = r"/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/data/all_nail/nail_image_preprocess"
    if not os.path.exists(filePath_bak):
        os.makedirs(filePath_bak)
    pic_YCrCb(filePath, filePath_bak)
    # print("xml files to bak...")
    # file_path_xml=os.path.join(filePath, '*.xml')
    # os.system('cp %s %s'% file_path_xml, filePath_bak)
