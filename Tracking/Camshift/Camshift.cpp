//
//  Camshift.cpp
//  Tracking
//
//  Created by FloodSurge on 8/5/15.
//  Copyright (c) 2015 FloodSurge. All rights reserved.
//

#include "Camshift.h"

void Camshift::initialize(const cv::Mat image, const cv::Rect objectBox)
{
    selection = objectBox;
    
    cvtColor(image, hsv, CV_BGR2HSV);
    
    int vmin = 10,vmax = 256, smin = 30;
    
    inRange(hsv, Scalar(0,smin,MIN(vmin, vmax)), Scalar(180,256,MAX(vmin, vmax)), mask);
    
    int ch[] = {0,0};
    hue.create(hsv.size(),hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);
    
    Mat roi(hue, selection), maskroi(mask, selection);
    float hranges[] = {0,180};
    const float* phranges = hranges;
    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
    normalize(hist, hist, 0, 255, NORM_MINMAX);

}

void Camshift::processFrame(const cv::Mat image)
{

    cvtColor(image, hsv, CV_BGR2HSV);
    
    int vmin = 10,vmax = 256, smin = 30;

    inRange(hsv, Scalar(0,smin,MIN(vmin, vmax)), Scalar(180,256,MAX(vmin, vmax)), mask);
    
    int ch[] = {0,0};
    hue.create(hsv.size(),hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);
    float hranges[] = {0,180};
    const float* phranges = hranges;
    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    backproj &= mask;
    objectBox = CamShift(backproj, selection,
                                    TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
    
}

Camshift::Camshift()
{
    
}



