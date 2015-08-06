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

#ifndef _OPENCV_color_tracker_HPP_
#define _OPENCV_color_tracker_HPP_ 

#include "color_tracker_parameters.h"

#ifdef __cplusplus
namespace cv {
	namespace colortracker {

		class ColorTracker
		{
			ColorTrackerParameters params;
			Mat im_patch;
			vector<vector<double> > w2c;
			vector<vector<double> > w2c_t; // transpose

			// running parameters
			bool use_dimensionality_reduction;
			cv::Size target_sz;
			cv::Point pos;
			// window size, taking padding into account
			cv::Size sz_with_padding;

			Mat yf;

			Mat cos_window;

			// initialize the projection matrix
			Mat projection_matrix;

			// appearance
			Mat z_npca, z_pca;
			Mat pca_variances, pca_basis;
			Mat old_cov_matrix;
			Mat alphaf_num, alphaf_den;

			Mat kf;
			Mat num1, num2, denum1;
			Mat response, tmp_r;
			vector<Mat> denum, num;

			Mat xo_npca, xo_pca;

			Mat data_mean, cov_matrix, data_matrix;
			// positions
			vector<pair<Point, Size> > positions;

			int frame_index;

			// functions
			vector<Mat> feature_projection(Mat &x_npca, Mat& x_pca, Mat& projection_matrix, Mat& cos_window);
			template<class T>
			vector<T> get_max(vector<vector<T> > &inp, int dim);
			vector<double> get_vector(int dim, int ind);
			Mat reshape(vector<double> &inp, int rows, int cols);
			vector<double> select_indeces(vector<double> &inp, vector<int> &indeces);
			vector<Mat> im2c(Mat &im, vector<vector<double> > &w2c, int color = 0);

			vector<Mat> get_feature_map(Mat &im_patch, vector<string> &features);

			void get_subwindow(Mat &im, cv::Point pos, cv::Size sz, vector<string> &non_pca_features, 
					vector<string> &pca_features, vector<vector<double> > &w2c,
				Mat &out_npca, Mat &out_pca);
			Mat mul_complex_element_by_element(Mat &xf, Mat &yf, int sign = 1);
			Mat mul_complex_element_by_element_second_conjugate(Mat &xf, Mat &yf);
			Mat dense_gauss_kernel(double sigma, vector<Mat>& x, vector<Mat>& y);
			void load_w2c();
		public:
			ColorTracker(ColorTrackerParameters &param);
			void init_tracking();
			cv::Rect track_frame(Mat &current_frame);
			void track_video(double start_second = 0, double end_second = 0);
			cv::Rect get_position();
		};
	}
}

#endif
#endif
