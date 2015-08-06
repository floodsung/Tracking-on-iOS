//
//  Camshift.h
//  Tracking
//
//  Created by FloodSurge on 8/5/15.
//  Copyright (c) 2015 FloodSurge. All rights reserved.
//

#ifndef __Tracking__Camshift__
#define __Tracking__Camshift__

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <vector>

using std::vector;
using namespace cv;

class Camshift {
    
    
public:
    Camshift();
    void initialize(const Mat image, const cv::Rect objectBox);
    void processFrame(const Mat image);
    RotatedRect objectBox;
private:
    Mat frame, hsv, hue, mask, hist, histimg, backproj;
    cv::Rect selection;
    int hsize = 16;
    
};

#endif /* defined(__Tracking__Camshift__) */
