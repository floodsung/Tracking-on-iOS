/* 
 * Struck: Structured Output Tracking with Kernels
 * 
 * Code to accompany the paper:
 *   Struck: Structured Output Tracking with Kernels
 *   Sam Hare, Amir Saffari, Philip H. S. Torr
 *   International Conference on Computer Vision (ICCV), 2011
 * 
 * Copyright (C) 2011 Sam Hare, Oxford Brookes University, Oxford, UK
 * 
 * This file is part of Struck.
 * 
 * Struck is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Struck is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Struck.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef STRUCK_TRACKER_H
#define STRUCK_TRACKER_H

#include "StruckRect.h"

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "Eigen/Core"

class Config;
class Features;
class Kernel;
class LaRank;
class ImageRep;

class StruckTracker
{
public:
	StruckTracker(const Config& conf);
	~StruckTracker();
	
	void Initialise(const cv::Mat& frame, FloatStruckRect bb);
	void Reset();
	void Track(const cv::Mat& frame);
	void Debug();
	
	inline const FloatStruckRect& GetBB() const { return m_bb; }
	inline bool IsInitialised() const { return m_initialised; }
	
private:
	const Config& m_config;
	bool m_initialised;
	std::vector<Features*> m_features;
	std::vector<Kernel*> m_kernels;
	LaRank* m_pLearner;
	FloatStruckRect m_bb;
	cv::Mat m_debugImage;
	bool m_needsIntegralImage;
	bool m_needsIntegralHist;
	
	void UpdateLearner(const ImageRep& image);
	void UpdateDebugImage(const std::vector<FloatStruckRect>& samples, const FloatStruckRect& centre, const std::vector<double>& scores);
};

#endif
