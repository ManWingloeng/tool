//
// Created by todd on 18-12-28.
//
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

bool find_hand(Mat& frame, Mat& img, Mat& bin_mask, Mat& res, vector<Point>& cmax) {
//    Mat img;
    bool status = false;
    frame.copyTo(img);
//    GaussianBlur(frame, frame, Size(15, 15), 0);
//    bilateralFilter(frame, frame, 9, 75, 100);
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    Mat YCrCb_frame;
    cvtColor(frame, YCrCb_frame, COLOR_BGR2YCrCb);
    Mat mask;
    inRange(YCrCb_frame, Scalar(0, 140, 100), Scalar(255, 170, 120), mask);
//    Mat res;
    bitwise_and(frame, frame, res, mask);
    Mat rect = getStructuringElement(MORPH_RECT, Size(15, 15));
//    dilate(res, res, rect, Point(-1, 1), 1);
//    erode(res, res, rect, Point(-1, 1), 1);
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

    drawContours(img, cmax, 0, Scalar(55, 55, 251), 1);
    return status;
}

