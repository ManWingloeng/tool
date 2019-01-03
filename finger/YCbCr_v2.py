# coidng:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import jit


@jit(nopython=True, parallel=True)
def gammaupdate(rows, cols, img, gamma):
    # for r in range(rows):
    #     for c in range(cols):
    #         # get values of blue, green, red
    #         # B = img.item(r,c,0)
    #         # G = img.item(r,c,1)
    #         # R = img.item(r,c,2)
    #         B = img[r][c][0]
    #         G = img[r][c][1]
    #         R = img[r][c][2]
    #         # gamma correction
    #         B = int(B ** gamma)
    #         G = int(G ** gamma)
    #         R = int(R ** gamma)
    #
    #         # set values of blue, green, red
    #         # img.itemset((r,c,0), B)
    #         # img.itemset((r,c,1), G)
    #         # img.itemset((r,c,2), R)
    #         img[r][c][0] = B
    #         img[r][c][1] = G
    #         img[r][c][2] = R
    np.power(img[:, :, 0], gamma)
    np.power(img[:, :, 1], gamma)
    np.power(img[:, :, 2], gamma)


@jit(nopython=True, parallel=True)
def Yupdate(rows, cols, imgYcc, Wcb,
            Wcr,
            WHCb,
            WHCr,
            WLCb,
            WLCr,
            Ymin,
            Ymax,
            Kl,
            Kh,
            WCb,
            WCr,
            CbCenter,
            CrCenter):
    for r in range(rows):
        for c in range(cols):

            # non-skin area if skin equals 0, skin area otherwise
            skin = 0

            ########################################################################

            # color space transformation

            # get values ycbcr color space
            # Y = imgYcc.item(r, c, 0)
            # Cr = imgYcc.item(r, c, 1)
            # Cb = imgYcc.item(r, c, 2)
            Y = imgYcc[r][c][0]
            Cr = imgYcc[r][c][1]
            Cb = imgYcc[r][c][2]
            if Y < Kl:
                WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin)
                WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin)

                CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
                CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)

            elif Y > Kh:
                WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh)
                WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh)

                CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
                CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)

            if Y < Kl or Y > Kh:
                imgYcc[r][c][1] = (Cr - CrCenter) * Wcr / WCr + 154
                imgYcc[r][c][2] = (Cb - CbCenter) * Wcb / WCb + 108
            ########################################################################

            # skin color detection

            # if Cb > 100 and Cb < 127 and Cr > 133 and Cr < 173:  # 77 127 133 173
            #     skin = 1
            #     # print 'Skin detected!'
            #
            # if 0 == skin:
            #     # imgSkin.itemset((r, c, 0), 0)
            #     # imgSkin.itemset((r, c, 1), 0)
            #     # imgSkin.itemset((r, c, 2), 0)
            #     imgSkin[r][c][0] = 0
            #     imgSkin[r][c][1] = 0
            #     imgSkin[r][c][2] = 0


# imgFile = 'images/face.jpg'
@jit(nogil=True)
def convertYCbCr2(frame):
    # load an original image
    img = frame.copy()
    # img = cv2.imread(imgFile)
    ################################################################################
    rows, cols, channels = img.shape
    ################################################################################
    # light compensation

    gamma = 0.95
    #
    # for r in range(rows):
    #     for c in range(cols):
    #         # get values of blue, green, red
    #         # B = img.item(r,c,0)
    #         # G = img.item(r,c,1)
    #         # R = img.item(r,c,2)
    #         B = img[r][c][0]
    #         G = img[r][c][1]
    #         R = img[r][c][2]
    #         # gamma correction
    #         B = int(B ** gamma)
    #         G = int(G ** gamma)
    #         R = int(R ** gamma)
    #
    #         # set values of blue, green, red
    #         # img.itemset((r,c,0), B)
    #         # img.itemset((r,c,1), G)
    #         # img.itemset((r,c,2), R)
    #         img[r][c][0] = B
    #         img[r][c][1] = G
    #         img[r][c][2] = R
    gammaupdate(rows, cols, img, gamma)
    ################################################################################

    # convert color space  rgb to ycbcr
    imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # convert color space  bgr to rgb                        
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # prepare an empty image space
    imgSkin = np.zeros(img.shape, np.uint8)
    # copy original image
    imgSkin = img.copy()
    ################################################################################

    # define variables for skin rules

    Wcb = 46.97
    Wcr = 38.76

    WHCb = 14
    WHCr = 10
    WLCb = 23
    WLCr = 20

    Ymin = 16
    Ymax = 235

    Kl = 125
    Kh = 188

    WCb = 0
    WCr = 0

    CbCenter = 0
    CrCenter = 0
    ############################################################################
    # for r in range(rows):
    #     for c in range(cols):
    #
    #         # non-skin area if skin equals 0, skin area otherwise
    #         skin = 0
    #
    #         ########################################################################
    #
    #         # color space transformation
    #
    #         # get values ycbcr color space
    #         # Y = imgYcc.item(r, c, 0)
    #         # Cr = imgYcc.item(r, c, 1)
    #         # Cb = imgYcc.item(r, c, 2)
    #         Y = imgYcc[r][c][0]
    #         Cr = imgYcc[r][c][1]
    #         Cb = imgYcc[r][c][2]
    #         if Y < Kl:
    #             WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin)
    #             WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin)
    #
    #             CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
    #             CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)
    #
    #         elif Y > Kh:
    #             WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh)
    #             WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh)
    #
    #             CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
    #             CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)
    #
    #         if Y < Kl or Y > Kh:
    #             Cr = (Cr - CrCenter) * Wcr / WCr + 154
    #             Cb = (Cb - CbCenter) * Wcb / WCb + 108
    #         ########################################################################
    #
    #         # skin color detection
    #
    #         if Cb > 77 and Cb < 127 and Cr > 133 and Cr < 173:
    #             skin = 1
    #             # print 'Skin detected!'
    #
    #         if 0 == skin:
    #             # imgSkin.itemset((r, c, 0), 0)
    #             # imgSkin.itemset((r, c, 1), 0)
    #             # imgSkin.itemset((r, c, 2), 0)
    #             imgSkin[r][c][0] = 0
    #             imgSkin[r][c][1] = 0
    #             imgSkin[r][c][2] = 0
    Yupdate(rows, cols, imgYcc, Wcb,
            Wcr,
            WHCb,
            WHCr,
            WLCb,
            WLCr,
            Ymin,
            Ymax,
            Kl,
            Kh,
            WCb,
            WCr,
            CbCenter,
            CrCenter)
    return imgYcc
@jit(nogil=True)
def detect_ellipse(frame):
    # load an original image
    img = frame.copy()
    rows, cols, channels = img.shape


    gamma = 0.95

    gammaupdate(rows, cols, img, gamma)
    ################################################################################

    # convert color space  rgb to ycbcr
    imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # convert color space  bgr to rgb                        
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # prepare an empty image space
    imgSkin = np.zeros(img.shape, np.uint8)
    # copy original image
    imgSkin = img.copy()
    ################################################################################

    # define variables for skin rules

    Wcb = 46.97
    Wcr = 38.76

    WHCb = 14
    WHCr = 10
    WLCb = 23
    WLCr = 20

    Ymin = 16
    Ymax = 235

    Kl = 125
    Kh = 188

    WCb = 0
    WCr = 0

    CbCenter = 0
    CrCenter = 0
    Yupdate(rows, cols, imgYcc, Wcb,
            Wcr,
            WHCb,
            WHCr,
            WLCb,
            WLCr,
            Ymin,
            Ymax,
            Kl,
            Kh,
            WCb,
            WCr,
            CbCenter,
            CrCenter)
    imgYcc_copy = imgYcc.copy()
    Cx, Cy = 109.38, 152.02
    ecx, ecy = 1.60, 2.41
    a, b = 25.39, 14.03
    Theta = 2.53 / np.pi * 180
    cv2.imshow("copy",imgSkin)
    for r in range(rows):
        for c in range(cols):
            Cb = imgYcc[r][c][2]
            Cr = imgYcc[r][c][1]
            cosTheta = np.cos(Theta)
            sinTehta = np.sin(Theta)
            matrixA = np.array([[cosTheta, sinTehta], [-sinTehta, cosTheta]], dtype=np.double)
            matrixB = np.array([[Cb - Cx], [Cr - Cy]], dtype=np.double)
            # 矩阵相乘
            matrixC = np.dot(matrixA, matrixB)
            x = matrixC[0, 0]
            y = matrixC[1, 0]
            ellipse = (x - ecx) ** 2 / a ** 2 + (y - ecy) ** 2 / b ** 2
            if ellipse > 1:
                imgSkin[r][c][0] = 0
                imgSkin[r][c][1] = 0
                imgSkin[r][c][2] = 0
    return imgSkin

