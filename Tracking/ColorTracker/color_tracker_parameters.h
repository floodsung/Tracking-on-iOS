/*
Copyright (c) 2015, Mostafa Mohamed (Izz)
izz.mostafa@gmail.com

All rights reserved.

Redistribution and use in source and binary forms, with or without modification
, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _OPENCV_color_tracker_parameters_HPP_
#define _OPENCV_color_tracker_parameters_HPP_ 

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>

#ifdef __cplusplus
namespace cv { namespace colortracker {


		//using namespace cv;
		using namespace std;

		class ColorTrackerParameters
		{
		public:
			//parameters according to the paper
			double padding;//params.padding = 1.0;         			   % extra area surrounding the target
			double output_sigma_factor;//params.output_sigma_factor = 1 / 16;		   % spatial bandwidth(proportional to target)
			double sigma;//params.sigma = 0.2;         			   % gaussian kernel bandwidth
			double lambda;//params.lambda = 1e-2;					   % regularization(denoted "lambda" in the paper)
			double learning_rate;//params.learning_rate = 0.075;			   % learning rate for appearance model update scheme(denoted "gamma" in the paper)
			double compression_learning_rate;//params.compression_learning_rate = 0.15;   % learning rate for the adaptive dimensionality reduction(denoted "mu" in the paper)
			vector<string> non_compressed_features;//params.non_compressed_features = { 'gray' }; % features that are not compressed, a cell with strings(possible choices : 'gray', 'cn')
			vector<string> compressed_features;//params.compressed_features = { 'cn' };       % features that are compressed, a cell with strings(possible choices : 'gray', 'cn')
			int num_compressed_dim;//params.num_compressed_dim = 2;             % the dimensionality of the compressed features
			//
			//params.visualization = 1;
			int visualization;
			string video_path;
			cv::Point init_pos;
			cv::Size wsize;

			ColorTrackerParameters();
		};

	}
}
#endif
#endif
