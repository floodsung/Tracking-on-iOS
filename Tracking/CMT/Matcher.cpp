#include "Matcher.h"

using cv::vconcat;
using cv::DMatch;

namespace cmt {

void Matcher::initialize(const vector<Point2f> & pts_fg_norm, const Mat desc_fg, const vector<int> & classes_fg,
        const Mat desc_bg, const Point2f center)
{

    //Copy normalized points 存储 正规化的点
    this->pts_fg_norm = pts_fg_norm;

    //Remember number of background points 存储背景的特征点的数量
    this->num_bg_points = desc_bg.rows;

    //Form database by stacking background and foreground features
    // 合成前景和背景的特征到一个Mat文件中
    if (desc_bg.rows > 0 && desc_fg.rows > 0)
        vconcat(desc_bg, desc_fg, database);
    else if (desc_bg.rows > 0)
        database = desc_bg;
    else
        database = desc_fg;

    //Extract descriptor length from features 根据特征抽取描述其长度
    desc_length = database.cols*8;

    //Create background classes (-1) 创建背景类
    vector<int> classes_bg = vector<int>(desc_bg.rows,-1);

    //Concatenate fg and bg classes 连接前景和背景的类
    classes = classes_bg;
    classes.insert(classes.end(), classes_fg.begin(), classes_fg.end());

    //Create descriptor matcher 创建描述匹配器BruteForce-Hamming
    bfmatcher = DescriptorMatcher::create("BruteForce-Hamming");

}

void Matcher::matchGlobal(const vector<KeyPoint> & keypoints, const Mat descriptors,
        vector<Point2f> & points_matched, vector<int> & classes_matched)
{
    //FILE_LOG(logDEBUG) << "Matcher::matchGlobal() call";

    if (keypoints.size() == 0)
    {
        //FILE_LOG(logDEBUG) << "Matcher::matchGlobal() return";
        return;
    }

    vector<vector<DMatch> > matches;
    bfmatcher->knnMatch(descriptors, database, matches, 2);

    for (size_t i = 0; i < matches.size(); i++)
    {
        vector<DMatch> m = matches[i];

        float distance1 = m[0].distance / desc_length;
        float distance2 = m[1].distance / desc_length;
        int matched_class = classes[m[0].trainIdx];

        if (matched_class == -1) continue;
        if (distance1 > thr_dist) continue;
        if (distance1/distance2 > thr_ratio) continue;

        points_matched.push_back(keypoints[i].pt);
        classes_matched.push_back(matched_class);
    }

    //FILE_LOG(logDEBUG) << "Matcher::matchGlobal() return";
}

void Matcher::matchLocal(const vector<KeyPoint> & keypoints, const Mat descriptors,
        const Point2f center, const float scale, const float rotation,
        vector<Point2f> & points_matched, vector<int> & classes_matched)
{

    if (keypoints.size() == 0) {
        return;
    }

    //Transform initial points
    vector<Point2f> pts_fg_trans;
    pts_fg_trans.reserve(pts_fg_norm.size());
    for (size_t i = 0; i < pts_fg_norm.size(); i++)
    {
        // 同样是计算相对位置
        pts_fg_trans.push_back(scale * rotate(pts_fg_norm[i], -rotation));
    }

    //Perform local matching
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        //Normalize keypoint with respect to center
        Point2f location_rel = keypoints[i].pt - center;

        //Find potential indices for matching
        vector<int> indices_potential;
        for (size_t j = 0; j < pts_fg_trans.size(); j++)
        {
            // 计算位置偏差
            float l2norm = norm(pts_fg_trans[j] - location_rel);

            // 设置一个阈值
            if (l2norm < thr_cutoff) {
                indices_potential.push_back(num_bg_points + j);
            }

        }

        //If there are no potential matches, continue
        if (indices_potential.size() == 0) continue;

        //Build descriptor matrix and classes from potential indices
        Mat database_potential = Mat(indices_potential.size(), database.cols, database.type());
        for (size_t j = 0; j < indices_potential.size(); j++) {
            database.row(indices_potential[j]).copyTo(database_potential.row(j));
        }

        //Find distances between descriptors
        vector<vector<DMatch> > matches;
        // 对选出的特征点进行特征匹配
        bfmatcher->knnMatch(descriptors.row(i), database_potential, matches, 2);

        vector<DMatch> m = matches[0];

        float distance1 = m[0].distance / desc_length;
        float distance2 = m.size() > 1 ? m[1].distance / desc_length : 1;

        if (distance1 > thr_dist) continue;
        if (distance1/distance2 > thr_ratio) continue;

        int matched_class = classes[indices_potential[m[0].trainIdx]];

        points_matched.push_back(keypoints[i].pt);
        classes_matched.push_back(matched_class);
    }

}

} /* namespace CMT */
