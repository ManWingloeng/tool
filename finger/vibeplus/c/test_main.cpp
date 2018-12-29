//
// Created by todd on 18-12-28.
//
#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
int main() {
    cout << "Hello, this is opencv tutorial." << std::endl;
    cv::Mat img = cv::imread("/home/todd/Pictures/tichu/筛选后的图片/豹/1.jpg");
    cv::namedWindow("Cat", CV_WINDOW_AUTOSIZE);
    cv::imshow("Cat", img);
    cv::waitKey(0);
    return 0;
}
