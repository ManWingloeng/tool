#include "ViBePlus.h"
#include "iostream"
#include "stdio.h"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

int getMaxAreaContourId(vector<vector<Point>> contours) {
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;
        }
    }
    return maxAreaContourId;
}

bool find_hand(Mat &frame, Mat &img, Mat &bin_mask, Mat &res, vector<Point> &cmax) {
//    Mat img;
    bool status = false;
    frame.copyTo(img);
    Mat Gframe;
//    GaussianBlur(frame, Gframe, Size(15, 15), 0);
    Mat Bframe;
//    bilateralFilter(Gframe, Bframe, 9, 75, 100);
//    frame = Bframe;
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    Mat YCrCb_frame;
    cvtColor(frame, YCrCb_frame, COLOR_BGR2YCrCb);
    Mat mask;
    inRange(YCrCb_frame, Scalar(0, 140, 100), Scalar(255, 170, 120), mask);
//    Mat res;
    bitwise_and(frame, frame, res, mask);
    Mat rect = getStructuringElement(MORPH_RECT, Size(15, 15));
//    dilate(res, res, rect, Point(-1, 1), 4);
//    erode(res, res, rect, Point(-1, 1), 3);
//    dilate(res, res, rect, Point(-1, 1), 1);
//    Mat bin_mask;
    threshold(mask, bin_mask, 0, 255, THRESH_BINARY + THRESH_OTSU);
    dilate(bin_mask, bin_mask, rect, Point(-1, 1), 4);
    erode(bin_mask, bin_mask, rect, Point(-1, 1), 1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(bin_mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    int cid = getMaxAreaContourId(contours);
//    vector<Point> cmax;
    if (cid != -1) {
        status = true;
        cmax = contours.at(cid);
    }

//    drawContours(img, cmax, 0, Scalar(55, 55, 251), 1);
    return status;
}


int main(int argc, char *argv[]) {
    Mat frame, gray, SegModel, UpdateModel;
    VideoCapture capture;
    capture = VideoCapture(0);
    if (!capture.isOpened()) {
        printf("You should open the camera");
        return 0;
    }
//    if (!capture.isOpened()) {
//        capture = VideoCapture("../Video/Camera Road 01.avi");
//        if (!capture.isOpened()) {
//            capture = VideoCapture("../../Video/Camera Road 01.avi");
//            if (!capture.isOpened()) {
//                cout << "ERROR: Did't find this video!" << endl;
//                return 0;
//            }
//        }
//    }

    capture.set(CV_CAP_PROP_FRAME_WIDTH, 160);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
    if (!capture.isOpened()) {
        cout << "No camera or video input!" << endl;
        return -1;
    }

    // 程序运行时间统计变量
    // the Time Statistical Variable of Program Running Time
    double time;
    double start;

    ViBePlus vibeplus(20,2,20,1);
    bool count = true;

    while (1) {

        capture >> frame;
        if (frame.empty())
            continue;
        Mat image = frame.clone();
        Mat img, bin_mask, res;
        vector<Point> cmax;
        // 捕获图像
        Mat frame_resize;

        resize(frame, frame_resize, Size(640, 360));
        imshow("resize___frame", frame_resize);
        find_hand(frame_resize, img, bin_mask, res, cmax);
        Mat gray_res;
        imshow("res", res);
        cvtColor(res, gray_res, COLOR_BGR2GRAY);
        imshow("gray_res",gray_res);
        vibeplus.FrameCapture(res);

        start = static_cast<double>(getTickCount());
        vibeplus.Run();
        time = ((double) getTickCount() - start) / getTickFrequency() * 1000;
        cout << "Time of Update ViBe+ Background: " << time << "ms" << endl;

        SegModel = vibeplus.getSegModel();
        UpdateModel = vibeplus.getUpdateModel();
//			morphologyEx(SegModel, SegModel, MORPH_OPEN, Mat());
        imshow("SegModel", SegModel);
        imshow("UpdateModel", UpdateModel);
//        imshow("resizeframe", frame_resize);
//        imshow("input", frame);

        if (waitKey(10) == 27) {
            destroyAllWindows();
            break;
        }
    }

    return 0;
}
