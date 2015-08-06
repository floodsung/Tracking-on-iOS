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

#ifndef SAMPLE_H
#define SAMPLE_H

#include "ImageRep.h"
#include "StruckRect.h"

#include <vector>

class Sample
{
public:
	Sample(const ImageRep& image, const FloatStruckRect& roi) :
		m_image(image),
		m_roi(roi)
	{
	}
	
	inline const ImageRep& GetImage() const { return m_image; }
	inline const FloatStruckRect& GetROI() const { return m_roi; }

private:
	const ImageRep& m_image;
	FloatStruckRect m_roi;
};

class MultiSample
{
public:
	MultiSample(const ImageRep& image, const std::vector<FloatStruckRect>& StruckRects) :
		m_image(image),
		m_StruckRects(StruckRects)
	{
	}
	
	inline const ImageRep& GetImage() const { return m_image; }
	inline const std::vector<FloatStruckRect>& GetStruckRects() const { return m_StruckRects; }
	inline Sample GetSample(int i) const { return Sample(m_image, m_StruckRects[i]); }

private:
	const ImageRep& m_image;
	std::vector<FloatStruckRect> m_StruckRects;
};

#endif
