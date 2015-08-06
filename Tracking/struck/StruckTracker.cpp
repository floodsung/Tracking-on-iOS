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

#include "StruckTracker.h"
#include "Config.h"
#include "ImageRep.h"
#include "Sampler.h"
#include "Sample.h"
#include "GraphUtils/GraphUtils.h"

#include "HaarFeatures.h"
#include "RawFeatures.h"
#include "HistogramFeatures.h"
#include "MultiFeatures.h"

#include "Kernels.h"

#include "LaRank.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "Eigen/Core"

#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace Eigen;

StruckTracker::StruckTracker(const Config& conf) :
	m_config(conf),
	m_initialised(false),
	m_pLearner(0),
	m_debugImage(2*conf.searchRadius+1, 2*conf.searchRadius+1, CV_32FC1),
	m_needsIntegralImage(false)
{
	Reset();
}

StruckTracker::~StruckTracker()
{
	delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
}

void StruckTracker::Reset()
{
	m_initialised = false;
	m_debugImage.setTo(0);
	if (m_pLearner) delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
	m_features.clear();
	m_kernels.clear();
	
	m_needsIntegralImage = false;
	m_needsIntegralHist = false;
	
	int numFeatures = m_config.features.size();
	vector<int> featureCounts;
	for (int i = 0; i < numFeatures; ++i)
	{
		switch (m_config.features[i].feature)
		{
		case Config::kFeatureTypeHaar:
			m_features.push_back(new HaarFeatures(m_config));
			m_needsIntegralImage = true;
			break;			
		case Config::kFeatureTypeRaw:
			m_features.push_back(new RawFeatures(m_config));
			break;
		case Config::kFeatureTypeHistogram:
			m_features.push_back(new HistogramFeatures(m_config));
			m_needsIntegralHist = true;
			break;
		}
		featureCounts.push_back(m_features.back()->GetCount());
		
		switch (m_config.features[i].kernel)
		{
		case Config::kKernelTypeLinear:
			m_kernels.push_back(new LinearKernel());
			break;
		case Config::kKernelTypeGaussian:
			m_kernels.push_back(new GaussianKernel(m_config.features[i].params[0]));
			break;
		case Config::kKernelTypeIntersection:
			m_kernels.push_back(new IntersectionKernel());
			break;
		case Config::kKernelTypeChi2:
			m_kernels.push_back(new Chi2Kernel());
			break;
		}
	}
	
	if (numFeatures > 1)
	{
		MultiFeatures* f = new MultiFeatures(m_features);
		m_features.push_back(f);
		
		MultiKernel* k = new MultiKernel(m_kernels, featureCounts);
		m_kernels.push_back(k);		
	}
	
	m_pLearner = new LaRank(m_config, *m_features.back(), *m_kernels.back());
}
	

void StruckTracker::Initialise(const cv::Mat& frame, FloatStruckRect bb)
{
	m_bb = IntStruckRect(bb);
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	for (int i = 0; i < 1; ++i)
	{
		UpdateLearner(image);
	}
	m_initialised = true;
}

void StruckTracker::Track(const cv::Mat& frame)
{
	assert(m_initialised);
	
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	
	vector<FloatStruckRect> StruckRects = Sampler::PixelSamples(m_bb, 10);
    //cout << "Struck rects size:" << StruckRects.size() << endl;
	vector<FloatStruckRect> keptStruckRects;
	keptStruckRects.reserve(StruckRects.size());
	for (int i = 0; i < (int)StruckRects.size(); ++i)
	{
		if (!StruckRects[i].IsInside(image.GetStruckRect())) continue;
		keptStruckRects.push_back(StruckRects[i]);
	}
	
	MultiSample sample(image, keptStruckRects);
	
    
	vector<double> scores;
	m_pLearner->Eval(sample, scores);
	
	double bestScore = -DBL_MAX;
	int bestInd = -1;
	for (int i = 0; i < (int)keptStruckRects.size(); ++i)
	{
        //cout << "No." << i << scores[i] << endl;
		if (scores[i] > bestScore)
		{
			bestScore = scores[i];
			bestInd = i;
		}
	}
	
	UpdateDebugImage(keptStruckRects, m_bb, scores);
	
	if (bestInd != -1)
	{
		m_bb = keptStruckRects[bestInd];
        //cout << "Best ind" << bestInd << endl;
		UpdateLearner(image);
		
		//cout << "track score: " << bestScore << endl;

	}
}

void StruckTracker::UpdateDebugImage(const vector<FloatStruckRect>& samples, const FloatStruckRect& centre, const vector<double>& scores)
{
	double mn = VectorXd::Map(&scores[0], scores.size()).minCoeff();
	double mx = VectorXd::Map(&scores[0], scores.size()).maxCoeff();
	m_debugImage.setTo(0);
	for (int i = 0; i < (int)samples.size(); ++i)
	{
		int x = (int)(samples[i].XMin() - centre.XMin());
		int y = (int)(samples[i].YMin() - centre.YMin());
		m_debugImage.at<float>(m_config.searchRadius+y, m_config.searchRadius+x) = (float)((scores[i]-mn)/(mx-mn));
	}
}

void StruckTracker::Debug()
{
	imshow("tracker", m_debugImage);
	m_pLearner->Debug();
}

void StruckTracker::UpdateLearner(const ImageRep& image)
{
	// note these return the centre sample at index 0
	vector<FloatStruckRect> StruckRects = Sampler::RadialSamples(m_bb, 2*m_config.searchRadius, 5, 16);
	//vector<FloatStruckRect> StruckRects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);
	
	vector<FloatStruckRect> keptStruckRects;
	keptStruckRects.push_back(StruckRects[0]); // the true sample
	for (int i = 1; i < (int)StruckRects.size(); ++i)
	{
		if (!StruckRects[i].IsInside(image.GetStruckRect())) continue;
		keptStruckRects.push_back(StruckRects[i]);
	}
		
		
	//cout << keptStruckRects.size() << " samples" << endl;

		
	MultiSample sample(image, keptStruckRects);
	m_pLearner->Update(sample, 0);
}
