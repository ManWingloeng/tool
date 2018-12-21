# coding:utf-8
import cv2
import cv2 as cv
import numpy as np
# def processing(self, cv_image):
#     try:
#         # If the user has not selected a region, just return the image
#         crop_image = cv_image

#         blur = cv2.blur(crop_image, (3, 3), 0)

#         hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#         lower_range = np.array([2, 0, 0])
#         upper_range = np.array([16, 255, 255])

#         mask = cv2.inRange(hsv, lower_range, upper_range)

#         skinkernel = np.ones((5, 5))
#         dilation = cv2.dilate(mask, skinkernel, iterations=1)
#         erosion = cv2.erode(dilation, skinkernel, iterations=1)

#         iltered = cv2.GaussianBlur(erosion, (15, 15), 1)
#         ret, thresh = cv2.threshold(filtered, 127, 255, 0)

#         # Process any special keyboard commands
#         if self.keystroke != -1:
#             try:
#                 cc = chr(self.keystroke & 255).lower()
#                 if cc == 'c':
#                     # Clear the current keypoints
#                     keypoints = list()
#                     self.detect_box = None
#             except Exception as e:
#                 print
#                 e
#     except Exception as e:
#         print
#         e

#     return hand_img

def hand_finding(input_image):

    blur = cv2.blur(input_image, (3, 3))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
    filtered = cv2.medianBlur(dilation2, 5)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    median = cv2.medianBlur(dilation2, 5)
    ret, thresh = cv2.threshold(median, 127, 255, 0)

    return thresh


frame = cv2.imread("/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/data/all_nail/test/06.jpg")
hand=hand_finding(frame)
cv2.imshow("hand",hand)
cv2.waitKey(1000)
cv2.destroyAllWindows()