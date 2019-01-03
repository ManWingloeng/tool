//
// Created by todd on 19-1-3.
//

#include <iostream>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cvaux.h>
#include <highgui.h>
#define _USE_MATH_DEFINES

#include <math.h>


using namespace cv;
#if 1
class skin_detector
{
public:
    /*基于RGB范围的皮肤检测*/
    Mat RGB_detect(Mat& img)
    {
        /*
            R>95 AND G>40 B>20 AND MAX(R,G,B)-MIN(R,G,B)>15 AND ABS(R-G)>15 AND R>G AND R>B
                OR
            R>220 AND G>210 AND B>170 AND ABS(R-G)<=15 AND R>B AND G>B
        */
        Mat detect = img.clone();
        detect.setTo(0);
        if (img.empty() || img.channels() != 3)
        {
            return detect;
        }
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                uchar *p_detect = detect.ptr<uchar>(i, j);
                uchar *p_img = img.ptr<uchar>(i, j);
                if ((p_img[2] > 95 && p_img[1]>40 && p_img[0] > 20 &&
                     (MAX(p_img[0], MAX(p_img[1], p_img[2])) - MIN(p_img[0], MIN(p_img[1], p_img[2])) > 15) &&
                     abs(p_img[2] - p_img[1]) > 15 && p_img[2] > p_img[1] && p_img[1] > p_img[0]) ||
                    (p_img[2] > 200 && p_img[1] > 210 && p_img[0] > 170 && abs(p_img[2] - p_img[1]) <= 15 &&
                     p_img[2] > p_img[0] &&  p_img[1] > p_img[0]))
                {
                    p_detect[0] = p_img[0];
                    p_detect[1] = p_img[1];
                    p_detect[2] = p_img[2];
                }

            }

        }
        return detect;
    }

    /*HSV颜色空间H范围筛选法*/
    Mat HSV_detector(Mat& src)
    {
        Mat hsv_image;
        int h = 0;
        int s = 1;
        int v = 2;
        cvtColor(src, hsv_image, CV_BGR2HSV); //首先转换成到HSV空间
        Mat output_mask = Mat::zeros(src.size(), CV_8UC1);
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                uchar *p_mask = output_mask.ptr<uchar>(i, j);
                uchar *p_src = hsv_image.ptr<uchar>(i, j);
                if (p_src[h] >= 0 && p_src[h] <= 20 && p_src[s] >=48 && p_src[v] >=50)
                {
                    p_mask[0] = 255;
                }
            }
        }
        Mat detect;
        src.copyTo(detect, output_mask);;
        return detect;
    }

    /*YCrCb颜色空间Cr,Cb范围筛选法*/
    Mat YCrCb_detect(Mat & src)
    {
        Mat ycrcb_image;
        int Cr = 1;
        int Cb = 2;
        cvtColor(src, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间
        Mat output_mask = Mat::zeros(src.size(), CV_8UC1);
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                uchar *p_mask = output_mask.ptr<uchar>(i, j);
                uchar *p_src = ycrcb_image.ptr<uchar>(i, j);
                if (p_src[Cr] >= 133 && p_src[Cr] <= 173 && p_src[Cb] >= 77 && p_src[Cb] <= 127)
                {
                    p_mask[0] = 255;
                }
            }
        }
        Mat detect;
        src.copyTo(detect, output_mask);;
        return detect;

    }

    Mat gammaupdate(Mat& img, double gamma)
    {
        Mat imgcopy = img.clone();
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                uchar *p_imgcopy = imgcopy.ptr<uchar>(i, j);
                for(int k = 0; k < 3; k++)
                    p_imgcopy[k] = pow(p_imgcopy[k], gamma);
            }
        }
        return imgcopy;
    }


    Mat YCrCb_v2_process(Mat& src)
    {
        Mat ycrcb_image;
        Mat ycrcb_gamma = gammaupdate(src, 0.95);
        double Wcb = 46.97;
        double Wcr = 38.76;
        int WHCb = 14;
        int WHCr = 10;
        int WLCb = 23;
        int WLCr = 20;

        int Ymin = 16;
        int Ymax = 235;

        int Kl = 125;
        int Kh = 188;

        double WCb = 0;
        double WCr = 0;

        int CbCenter = 0;
        int CrCenter = 0;
        for (int i = 0; i < ycrcb_gamma.rows; i++) {
            for (int j = 0; j < ycrcb_gamma.cols; j++) {
                int skin = 0;

                uchar *p_imgYcc = ycrcb_gamma.ptr<uchar>(i, j);
                int Y = p_imgYcc[0];
                int Cr = p_imgYcc[1];
                int Cb = p_imgYcc[2];
                if (Y < Kl){
                    WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin);
                    WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin);

                    CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin);
                    CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin);
                }
                else if(Y > Kh) {
                    WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh);
                    WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh);

                    CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh);
                    CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh);
                }
                if (Y < Kl || Y > Kh){
                    p_imgYcc[1] = uchar((Cr - CrCenter) * Wcr / WCr + 154);
                    p_imgYcc[2] = uchar((Cb - CbCenter) * Wcb / WCb + 108);
                }

            }
        }
        return ycrcb_gamma;
    }
    /*YCrCb颜色空间Cr分量+Otsu法*/
    Mat YCrCb_Otsu_detect(Mat& src)
    {
        Mat ycrcb_image;
        cvtColor(src, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间
        Mat detect;
        vector<Mat> channels;
        split(ycrcb_image, channels);
        Mat output_mask = channels[1];
        threshold(output_mask, output_mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        src.copyTo(detect, output_mask);
        return detect;

    }

    /*基于椭圆皮肤模型的皮肤检测*/
    Mat ellipse_detect(Mat& src)
    {
        Mat img = src.clone();
        Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);
        //利用opencv自带的椭圆生成函数先生成一个肤色椭圆模型
        ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);
        Mat ycrcb_image;
        Mat output_mask = Mat::zeros(img.size(), CV_8UC1);
        cvtColor(img, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间
        for (int i = 0; i < img.cols; i++)   //利用椭圆皮肤模型进行皮肤检测
            for (int j = 0; j < img.rows; j++)
            {
                Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
                if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)   //如果该落在皮肤模型椭圆区域内，该点就是皮肤像素点
                    output_mask.at<uchar>(j, i) = 255;
            }

        Mat detect;
        img.copyTo(detect,output_mask);  //返回肤色图
        return detect;
    }

    Mat ellipse_detect_v2(Mat& src)
    {
        Mat img = src.clone();
        Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);

        double Cx = 109.38, Cy = 152.02;
        double ecx = 1.60, ecy = 2.41;
        double a = 25.39, b = 14.03;
        double Theta = 2.53 / M_PI * 180;
        //利用opencv自带的椭圆生成函数先生成一个肤色椭圆模型


        ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);
        Mat ycrcb_image;

        Mat output_mask = Mat::zeros(img.size(), CV_8UC1);
        cvtColor(img, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间
        Mat ycc_2 = YCrCb_v2_process(ycrcb_image);
        for (int i = 0; i < img.cols; i++)   //利用椭圆皮肤模型进行皮肤检测
            for (int j = 0; j < img.rows; j++)
            {
                Vec3b ycrcb = ycc_2.at<Vec3b>(j, i);
                if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)   //如果该落在皮肤模型椭圆区域内，该点就是皮肤像素点
                    output_mask.at<uchar>(j, i) = 255;
            }

        Mat detect;
        img.copyTo(detect,output_mask);  //返回肤色图
        return detect;
    }

    /*opencv自带肤色检测类AdaptiveSkinDetector*/
    Mat AdaptiveSkinDetector_detect(Mat& src)
    {
//        Mat* matp = &src;
//        IplImage *frame;
        IplImage* frame = cvCreateImage(cvSize(src.cols, src.rows), IPL_DEPTH_8U, 3);
        IplImage ipltemp = src;
        cvCopy(&ipltemp, frame);
//        frame = &IplImage(&src);  //Mat -> IplImage
        CvAdaptiveSkinDetector filter(1, CvAdaptiveSkinDetector::MORPHING_METHOD_ERODE_DILATE);

        IplImage *maskImg = cvCreateImage(cvSize(src.cols, src.rows), IPL_DEPTH_8U, 1);
        IplImage *skinImg = cvCreateImage(cvSize(src.cols, src.rows), IPL_DEPTH_8U, 3);
        cvZero(skinImg);
        filter.process(frame, maskImg);    // process the frame
        cvCopy(frame, skinImg, maskImg);
        Mat tmp(skinImg);  //IplImage -> Mat
        Mat detect = tmp.clone();
        cvReleaseImage(&skinImg);
        cvReleaseImage(&maskImg);
        return detect;
    }
};
#endif




int main(int argc, char** argv)
{

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        printf("fail to open camera!\n");
        return -1;
    }
    Mat frame;
    skin_detector detector;

    while (1)
    {
        cap >> frame;
        if (frame.empty())
        {
            continue;
        }
        Mat skin = detector.AdaptiveSkinDetector_detect(frame);
        Mat ellipse = detector.ellipse_detect(frame);
        Mat HSV = detector.HSV_detector(frame);
        imshow("AdaptiveSkinDetector capture skin", skin);
        imshow("YCbCr ellipse", ellipse);
        imshow("HSV",HSV);
        imshow("capture src",frame);

        if (waitKey(1) == 27)
        {
            break;
        }
    }

    return 0;
}