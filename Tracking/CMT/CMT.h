#ifndef CMT_H

#define CMT_H

#include "common.h"
#include "Consensus.h"
#include "Fusion.h"
#include "Matcher.h"
#include "Tracker.h"

#include <opencv2/features2d/features2d.hpp>

using cv::FeatureDetector;
using cv::DescriptorExtractor;

using cv::RotatedRect;
using cv::Size2f;

namespace cmt
{

class CMT
{
public:
    CMT() : str_detector("FAST"), str_descriptor("BRISK") {};
    void initialize(const Mat im_gray, const cv::Rect rect);
    void processFrame(const Mat im_gray);

    Fusion fusion;   // 数据融合器
    Matcher matcher; // 特征匹配器
    Tracker tracker; // 跟踪器
    Consensus consensus;  // 一致检查器

    string str_detector;
    string str_descriptor;

    vector<Point2f> points_active; //public for visualization purposes 有效特征点
    RotatedRect bb_rot;

private:
    cv::Ptr<FeatureDetector> detector;
    cv::Ptr<DescriptorExtractor> descriptor;

    Size2f size_initial;  // 初始大小

    vector<int> classes_active;

    float theta;

    Mat im_prev;
};

} /* namespace CMT */

#endif /* end of include guard: CMT_H */
