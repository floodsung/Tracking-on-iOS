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
#include "color_tracker.h"


using namespace cv;
using namespace cv::colortracker;

void ColorTracker::load_w2c()
{
	// load the normalized Color Name matrix
	ifstream ifstr("w2crs.txt");
	for (int i = 0; i < 10; i++)
	{
		w2c_t.push_back(vector<double>(32768,0));
	}
	vector<double> tmp(10, 0);
	for (int i = 0; i < 32768; i++)
	{
		w2c.push_back(tmp);
	}
	double tmp_val;
	for (int i = 0; i < 32768; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			ifstr >> tmp_val;
			w2c[i][j] = w2c_t[j][i] = tmp_val;
		}
	}
	ifstr.close();
}

ColorTracker::ColorTracker(ColorTrackerParameters &param)
{
	if (w2c.size() == 0)
	{
        cout << "load w2c" << endl;
		load_w2c();
	}
	cout << w2c.size() << endl;
	params = param;
}
vector<Mat> ColorTracker::feature_projection(Mat &x_npca, Mat& x_pca, Mat& projection_matrix, Mat& cos_window)
{
	// Calculates the compressed feature map by mapping the PCA features with
	// the projection matrix and concatinates this with the non-PCA features.
	// The feature map is then windowed.

	vector<Mat> z;

	if (x_pca.cols == 0)
	{
		// if no PCA-features exist, only use non-PCA
		z.push_back(x_npca);
	}
	else
	{
		// add x_npca if it exists first
		if (x_npca.cols != 0)
		{ //if not empty
			z.push_back(x_npca);
		}

		// project the PCA-features using the projection matrix and reshape
		// to a window
		Mat tmp = (x_pca * projection_matrix);
		int sizes[] = { cos_window.rows, cos_window.cols, projection_matrix.cols };
		for (int i = 0; i < tmp.cols; i++)
		{
			Mat tmpCol = tmp.col(i).clone();
			Mat tmpCol2(cos_window.cols, cos_window.rows, CV_64FC1);
			memcpy(tmpCol2.data, tmpCol.data, tmp.rows * sizeof(double));
			// concatinate the feature windows
			z.push_back(tmpCol2.t());
		}
	}

	// do the windowing of the output
	int sz = z.size();
	for (int i = 0; i < sz; i++)
	{
		Mat tmp = z[i].mul(cos_window);
		z[i] = tmp;
	}
	return z;
}

template<class T>
vector<T> ColorTracker::get_max(vector<vector<T> > &inp, int dim)
{
	// dim = 1 max row, 2 max column
	vector<T> ret;
	if (dim == 1)
	{
		int inp0_size = inp[0].size();
		for (int j = 0; j < inp0_size; j++)
		{
			ret.push_back(inp[0][j]);
			int inp_size = inp.size();
			for (int i = 1; i < inp_size; i++)
			{
				if (inp[i][j] > ret[j])
				{
					ret[j] = inp[i][j];
				}
			}
		}
	}
	else if (dim == 2)
	{
		int inp_size = inp.size();
		for (int i = 0; i < inp_size; i++)
		{
			ret.push_back(*max_element(inp[i].begin(), inp[i].end()));
		}
	}
	return ret;
}
vector<double> ColorTracker::get_vector(int dim, int ind)
{
	// dim = 1  row, 2  column
	if (dim == 2)
	{
		return w2c_t[ind];
	}
	else// if (dim == 1)
	{
		return w2c[ind];
	}
}
Mat ColorTracker::reshape(vector<double> &inp, int rows, int cols)
{
	Mat result(rows, cols, CV_64FC1);
	double* data = ((double*)result.data);
	memcpy(data, ((double*)(&inp[0])), rows*cols*sizeof(double));
	return result;
}
vector<double> ColorTracker::select_indeces(vector<double> &inp, vector<int> &indeces)
{
	int sz = std::min(inp.size(), indeces.size());
	vector<double> res(sz, 0);
	for (int i = 0; i < sz; i++)
	{
		res[i] = inp[indeces[i]];
	}
	return res;
}
vector<Mat> ColorTracker::im2c(Mat &im, vector<vector<double> > &w2c, int color)
{
	vector<Mat> out;
	// input im should be DOUBLE !
	// color=0 is color names out
	// color=-1 is colored image with color names out
	// color=1-11 is prob of colorname=color out;
	// color=-1 return probabilities
	// order of color names: black ,   blue   , brown       , grey       , green   , orange   , pink     , purple  , red     , white    , yellow
	double color_values[][3] = { { 0, 0, 0 }, { 0, 0, 1 }, { .5, .4, .25 }, { .5, .5, .5 }, { 0, 1, 0 }, { 1, .8, 0 }, { 1, .5, 1 }
	, { 1, 0, 1 }, { 1, 0, 0 }, { 1, 1, 1 }, { 1, 1, 0 } };

	vector<Mat> im_split;
	cv::split(im, im_split);
	Mat RR = im_split[2];
	Mat GG = im_split[1];
	Mat BB = im_split[0];
	
	double*  RRdata = ((double*)RR.data), *GGdata = ((double*)GG.data), *BBdata = ((double*)BB.data);
	int w = RR.cols;
	int h = RR.rows;
	vector<int> index_im(w * h, 0);
	int l = index_im.size();

	for (int i = 0; i < l; i++)
	{
		//int j = (i / w) + (i//w) * h;
		// I don't need +1 in the formula because the indeces are zero based here
		index_im[i] = (int)(floor(RRdata[i] / 8) + 32 * floor(GGdata[i] / 8) + 32 * 32 * floor(BBdata[i] / 8));
	}

	if (color == 0)
	{
		vector<double> w2cM = get_max(w2c, 2);
		vector<double> selected = select_indeces(w2cM, index_im);
		out.push_back(reshape(selected, im.rows, im.cols));
	}

	if (color > 0 && color < 12)
	{
		vector<double> w2cM = get_vector(2, color - 1);
		vector<double> selected = select_indeces(w2cM, index_im);
		out.push_back(reshape(selected, im.rows, im.cols));
	}

	if (color == -1)
	{
		out.push_back(im);
		vector<double> w2cM = get_max(w2c, 2);
		vector<double> temp = select_indeces(w2cM, index_im);
		Mat out2 = reshape(temp, im.rows, im.cols);
	}

	if (color == -2)
	{
		for (int i = 0; i < 10; i++)
		{
			vector<double> vec = get_vector(2, i);
			vector<double> selected = select_indeces(vec, index_im);
			Mat temp = reshape(selected, im.rows, im.cols);
			out.push_back(temp);
		}
	}

	return out;
}

vector<Mat> ColorTracker::get_feature_map(Mat &im_patch, vector<string> &features)
{
	// the names of the features that can be used
	vector<string> valid_features({ "gray", "cn" });
	//
	// the dimension of the valid features
	vector<int> feature_levels({ 1, 10 });

	int num_valid_features = valid_features.size();
	//cout << "here" << endl;
	vector<bool> used_features(num_valid_features, false);
	// get the used features
	for (int i = 0; i < num_valid_features; i++)
	{
		if (find(features.begin(), features.end(), valid_features[i]) != features.end())
		{
			used_features[i] = 1;
		}
	}


	// total number of used feature levels
	int num_feature_levels = 0;
	int feature_levels_size = feature_levels.size();
	for (int i = 0; i < feature_levels_size; i++)
	{
		num_feature_levels += feature_levels[i] * used_features[i];
	}
	int level = 0;
	// allocate space (for speed)
	vector<Mat> out(num_feature_levels, Mat::zeros(im_patch.rows, im_patch.cols, CV_64FC1));
	// If grayscale image
	if (im_patch.channels() == 1)
	{
		// Features that are available for grayscale sequances
		// Grayscale values (image intensity)
		im_patch.convertTo(out[0], CV_64F);
		out[0] = (out[0] / 255) - 0.5;
	}
	else
	{

		// Features that are available for color sequances

		// Grayscale values (image intensity)
		if (used_features[0])
		{
			cv::cvtColor(im_patch, out[0], CV_BGR2GRAY);
			out[0].convertTo(out[0], CV_64F);
			out[0] = (out[0] / 255) - 0.5;
		}

		// Color Names
		if (used_features[1])
		{
			// extract color descriptor
			Mat double_patch;
			im_patch.convertTo(double_patch, CV_64FC1);
			out = im2c(double_patch, w2c, -2);
		}
	}

	return out;
}

void ColorTracker::get_subwindow(Mat &im, cv::Point pos, cv::Size sz, vector<string> &non_pca_features, vector<string> &pca_features, vector<vector<double> > &w2c,
	Mat &out_npca, Mat &out_pca)
{
	// Extracts the non-PCA and PCA features from image im at position pos and
	// window size sz. The features are given in non_pca_features and
	// pca_features. out_npca is the window of non-PCA features and out_pca is
	// the PCA-features reshaped. 
	// w2c is the Color Names matrix if used.

	cv::Rect range((int)floor(pos.x - sz.width / 2), (int)floor(pos.y - sz.height / 2),
		0, 0);
	range.width = sz.width + range.x;
	range.height = sz.height + range.y;
	//check for out-of-bounds coordinates, and set them to the values at
	//the borders
	int top = 0, bottom = 0, left = 0, right = 0;

	if (range.width > (im.cols - 1))
	{
		right = (range.width - im.cols + 1);
		range.width = im.cols - 1;
	}
	if (range.height > (im.rows - 1))
	{
		bottom = (range.height - im.rows + 1);
		range.height = im.rows - 1;
	}
	if (range.x < 0)
	{
		left = -range.x;
		range.x = 0;
	}
	if (range.y < 0)
	{
		top = -range.y;
		range.y = 0;
	}
	range.width -= (range.x);
	range.height -= (range.y);

	//extract image
	// Mat im_patch(sz.height,sz.width,im.type());
	copyMakeBorder(im(range).clone(), im_patch, top, bottom, left, right, BORDER_REPLICATE);

	// compute non-pca feature map
	if (non_pca_features.size())
	{
		out_npca = get_feature_map(im_patch, non_pca_features)[0];
	}

	// compute pca feature map
	if (pca_features.size())
	{
		vector<Mat> temp_pca = get_feature_map(im_patch, pca_features);
		int total_len = sz.width*sz.height;
		out_pca = Mat::zeros(temp_pca.size(), total_len, CV_64FC1);
		int ind = 0;
		double* data = ((double*)out_pca.data);
		int temp_pca_size = temp_pca.size();
		for (int i = 0; i < temp_pca_size; i++)
		{
			Mat tmp = temp_pca[i].t();
			memcpy(data + i * total_len, tmp.data, total_len * sizeof(double));
		}
		out_pca = out_pca.t();
	}
}
Mat ColorTracker::mul_complex_element_by_element(Mat &xf, Mat &yf, int sign)
{
	Mat tmp_xf[2], tmp_yf[2];
	cv::split(xf, tmp_xf);
	cv::split(yf, tmp_yf);
	vector<Mat> tmp1;
	Mat k1 = tmp_xf[0].mul(tmp_yf[0] + sign * tmp_yf[1]);
	Mat k2 = sign * tmp_yf[1].mul(tmp_xf[0] + tmp_xf[1]);
	Mat k3 = tmp_yf[0].mul(tmp_xf[1] - tmp_xf[0]);
	tmp1.push_back(k1 - k2);
	tmp1.push_back(k1 + k3);
	Mat tmp2;
	cv::merge(tmp1, tmp2);

	return tmp2;
}

Mat ColorTracker::mul_complex_element_by_element_second_conjugate(Mat &xf, Mat &yf)
{
	return mul_complex_element_by_element(xf, yf, -1);
}

Mat ColorTracker::dense_gauss_kernel(double sigma, vector<Mat>& x, vector<Mat>& y)
{
	Mat res;
	// Computes the kernel output for multi-dimensional feature maps x and y
	// using a Gaussian kernel with standard deviation sigma.
	// xf = fft2(x);  //x in Fourier domain
	double xx = 0;
	vector<Mat> xf;
	int x_size = x.size();
	for (int i = 0; i < x_size; i++)
	{
		Mat tmp_xf;
		cv::dft(x[i], tmp_xf,DFT_COMPLEX_OUTPUT);
		xf.push_back(tmp_xf);
		int sz = x[0].rows * x[0].cols;
		double* a = ((double*)x[i].data);
		for (int j = 0; j < sz; j++, a++)
		{
			xx += ((*a) * (*a));
		}
	}


	double yy = 0;
	vector<Mat> yf;
	if (y.size())//general case, x and y are different
	{
		int y_size = y.size();
		for (int i = 0; i < y_size; i++)
		{
			Mat tmp_yf;
			cv::dft(y[i], tmp_yf,DFT_COMPLEX_OUTPUT);
			yf.push_back(tmp_yf);
			int sz = y[0].rows * y[0].cols;
			double* a = ((double*)y[i].data);
			for (int j = 0; j < sz; j++, a++)
			{
				yy += ((*a) * (*a));
			}
		}
	}
	else
	{
		//auto-correlation of x, avoid repeating a few operations
		yf = xf;
		yy = xx;
	}

	//cross-correlation term in Fourier domain
	vector<Mat> xyf;
	int xf_size = xf.size();
	for (int i = 0; i < xf_size; i++)
	{
		xyf.push_back(mul_complex_element_by_element_second_conjugate(xf[i], yf[i]));
	}
	Mat sum_xyf = xyf[0];
	int xyf_size = xyf.size();
	for (int i = 1; i < xyf_size; i++)
	{
		sum_xyf += xyf[i];
	}
	Mat xy;
	vector<Mat> v;
	dft(sum_xyf, xy, DFT_INVERSE);
	xy = xy / (xy.cols * xy.rows);

	split(xy, v);
	Mat real = v[0];

	//calculate gaussian response for all positions
	cv::exp(((-1 / (sigma * sigma)) * max(0, (xx + yy - 2 * real) / (x[0].rows * x[0].cols * x.size()))), res);
	return res;
}

void ColorTracker::init_tracking()
{
	// use_dimensionality_reduction
	use_dimensionality_reduction = (params.compressed_features.size() != 0);

	target_sz = cv::Size(params.wsize.width, params.wsize.height);
	pos = cv::Point(params.init_pos.x, params.init_pos.y);

	// window size, taking padding into account
	sz_with_padding = cv::Size((int)floor(target_sz.width * (1 + params.padding)), (int)floor(target_sz.height * (1 + params.padding)));

	// desired output(gaussian shaped), bandwidth proportional to target size
	double output_sigma = sqrt(target_sz.width * target_sz.height) * params.output_sigma_factor;
	Mat y = Mat::zeros(sz_with_padding, CV_64FC1);
	double output_sigma_square = output_sigma * output_sigma;
	int ind = 0;
	double *data = ((double*)(y.data));
	for (int i = 1; i <= sz_with_padding.height; i++)
	{
		for (int j = 1; j <= sz_with_padding.width; j++)
		{
			int tmpCs = j - sz_with_padding.width / 2;
			int tmpRs = i - sz_with_padding.height / 2;
			data[ind++] = exp(-0.5 / output_sigma_square * (tmpRs * tmpRs + tmpCs * tmpCs));
		}
	}

	cv::dft(y, yf, DFT_COMPLEX_OUTPUT);

	// store pre - computed cosine window
	Mat hann_1 = Mat::zeros(cv::Size(1, sz_with_padding.height), CV_64FC1);
	ind = 0;
	data = ((double*)hann_1.data);
	for (int i = 0; i < sz_with_padding.height; i++)
	{
		data[ind++] = (0.5 * (1 - cos(2 * 3.14159265359 * i / (sz_with_padding.height - 1))));
	}
	Mat hann_2 = Mat::zeros(cv::Size(1, sz_with_padding.width), CV_64FC1);
	ind = 0;
	data = ((double*)hann_2.data);
	for (int i = 0; i < sz_with_padding.width; i++)
	{
		data[ind++] = (0.5 * (1 - cos(2 * 3.14159265359 * i / (sz_with_padding.width - 1))));
	}

	cos_window = hann_1 * hann_2.t();

	frame_index = 1;
}

cv::Rect ColorTracker::track_frame(Mat &current_frame)
{
	if (frame_index > 1)
	{
		// compute the compressed learnt appearance
		vector<Mat> zp = feature_projection(z_npca, z_pca, projection_matrix, cos_window);

		// extract the feature map of the local image patch
		Mat xo_npca, xo_pca;
		get_subwindow(current_frame, pos, sz_with_padding, params.non_compressed_features, params.compressed_features, w2c, xo_npca, xo_pca);

		// do the dimensionality reduction and windowing
		vector<Mat> x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);

		// calculate the response of the classifier
		cv::dft(dense_gauss_kernel(params.sigma, x, zp), kf, DFT_COMPLEX_OUTPUT);
		num1 = mul_complex_element_by_element(alphaf_num, kf);
		num2 = mul_complex_element_by_element_second_conjugate(num1, alphaf_den);
		denum1 = mul_complex_element_by_element_second_conjugate(alphaf_den, alphaf_den);
		cv::split(denum1, denum);
		cv::split(num2, num);
		cv::divide(num[0], denum[0], num[0]);
		cv::divide(num[1], denum[0], num[1]);
		cv::merge(num, tmp_r);

		cv::dft(tmp_r, response, DFT_INVERSE | DFT_REAL_OUTPUT);
		response = response / (response.cols * response.rows);

		// target location is at the maximum response
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		cv::minMaxLoc(response, &minVal, &maxVal, &minLoc, &maxLoc);
		pos = pos - cv::Point((int)floor(sz_with_padding.width / 2), (int)floor(sz_with_padding.height / 2))
			+ cv::Point(maxLoc.x + 1, maxLoc.y + 1);
		if (pos.x < 0)
		{
			pos.x = 0;
		}
		if (pos.y < 0)
		{
			pos.y = 0;
		}
		if (pos.x >= current_frame.cols)
		{
			pos.x = current_frame.cols - 1;
		}
		if (pos.y >= current_frame.rows)
		{
			pos.y = current_frame.rows;
		}
	}
	else
	{
		im_patch = Mat::zeros(sz_with_padding.height, sz_with_padding.width, current_frame.type());
	}

	// extract the feature map of the local image patch to train the classifer
	get_subwindow(current_frame, pos, sz_with_padding, params.non_compressed_features,
		params.compressed_features, w2c, xo_npca, xo_pca);

	if (frame_index == 1)
	{
		// initialize the appearance
		z_npca = xo_npca;
		z_pca = xo_pca;

		// set number of compressed dimensions to maximum if too many
		params.num_compressed_dim = std::min(params.num_compressed_dim, xo_pca.cols);
	}
	else
	{
		// update the appearance
		z_npca = (1 - params.learning_rate) * z_npca + params.learning_rate * xo_npca;
		z_pca = (1 - params.learning_rate) * z_pca + params.learning_rate * xo_pca;
	}

	// if dimensionality reduction is used: update the projection matrix
	if (use_dimensionality_reduction)
	{
		if (frame_index == 1)
		{
			// compute the mean appearance
			data_matrix = Mat::zeros(z_pca.rows, z_pca.cols, CV_64FC1);
		}
		// compute the mean appearance
		reduce(z_pca, data_mean, 0, CV_REDUCE_AVG);

		// substract the mean from the appearance to get the data matrix
		double*data = ((double*)data_matrix.data);
		for (int i = 0; i < z_pca.rows; i++)
		{
			memcpy(data + i * z_pca.cols, ((Mat)(z_pca.row(i) - data_mean)).data, z_pca.cols * sizeof(double));
		}

		// calculate the covariance matrix
		cov_matrix = (1.0 / (sz_with_padding.width * sz_with_padding.height - 1))
			* (data_matrix.t() * data_matrix);
		//cov_matrix.convertTo(cov_matrix, CV_32FC1);

		// calculate the principal components (pca_basis) and corresponding variances
		if (frame_index == 1)
		{
			Mat vt;
			cv::SVD::compute(cov_matrix, pca_variances, pca_basis, vt);
		}
		else
		{
			Mat vt;
			cv::SVD::compute((1 - params.compression_learning_rate) * old_cov_matrix + params.compression_learning_rate * cov_matrix,
				pca_variances, pca_basis, vt);
		}

		// calculate the projection matrix as the first principal
		// components and extract their corresponding variances
		projection_matrix = pca_basis(cv::Rect(0, 0, params.num_compressed_dim, pca_basis.rows)).clone();
		Mat projection_variances = Mat::zeros(params.num_compressed_dim, params.num_compressed_dim, CV_64FC1);
		for (int i = 0; i < params.num_compressed_dim; i++)
		{
			((double*)projection_variances.data)[i + i*params.num_compressed_dim] = ((double*)pca_variances.data)[i];
		}

		if (frame_index == 1)
		{
			// initialize the old covariance matrix using the computed
			// projection matrix and variances
			old_cov_matrix = projection_matrix * projection_variances * projection_matrix.t();
		}
		else
		{
			// update the old covariance matrix using the computed
			// projection matrix and variances
			old_cov_matrix =
				(1 - params.compression_learning_rate) * old_cov_matrix +
				params.compression_learning_rate * (projection_matrix * projection_variances * projection_matrix.t());
		}
	}


	// project the features of the new appearance example using the new
	// projection matrix
	vector<Mat> x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);

	// calculate the new classifier coefficients
	vector<Mat> dummy;
	Mat tmpKernel = dense_gauss_kernel(params.sigma, x,dummy);
	Mat v[2];

	cv::dft(tmpKernel, kf, DFT_COMPLEX_OUTPUT);
	Mat new_alphaf_num = mul_complex_element_by_element(yf, kf);
	vector<Mat> kf_;
	cv::split(kf, kf_);
	kf_[0] += params.lambda;
	Mat tmp;
	cv::merge(kf_, tmp);
	Mat new_alphaf_den = mul_complex_element_by_element(kf, tmp);

	if (frame_index == 1)
	{
		// first frame_index, train with a single image
		alphaf_num = new_alphaf_num;
		alphaf_den = new_alphaf_den;
	}
	else
	{
		// subsequent frame_indexs, update the model
		alphaf_num = (1 - params.learning_rate) * alphaf_num + params.learning_rate * new_alphaf_num;
		alphaf_den = (1 - params.learning_rate) * alphaf_den + params.learning_rate * new_alphaf_den;
	}

	//save position
	positions.push_back(make_pair(pos - Point(1, 1), target_sz));

	//visualization
	if (params.visualization == 1)
	{
		cv::Rect rect(pos.x - 1 - target_sz.width / 2, pos.y - 1 - target_sz.height / 2, target_sz.width, target_sz.height);
		cv::rectangle(current_frame, rect, Scalar(0, 0, 255), 2);
		imshow("current_frame", current_frame);
		cv::waitKey(30);
	}
	frame_index++;

	return get_position();
}

cv::Rect ColorTracker::get_position()
{
	cv::Rect rect(pos.x - 1 - target_sz.width / 2, pos.y - 1 - target_sz.height / 2, target_sz.width, target_sz.height);
	return rect;
}

void ColorTracker::track_video(double start_second,double end_second)
{
	init_tracking();

	Mat current_frame;
	// read the video
	VideoCapture vcap;
	if (vcap.open(params.video_path))
	{
		int numberOfFrames = vcap.get(CAP_PROP_FRAME_COUNT); // get frame count
		double framerate = vcap.get(CAP_PROP_FPS); //get the frame rate
		framerate = 30;		
		//cout << framerate << endl;		
		unsigned int start_frame = start_second * framerate;
		unsigned int end_frame = end_second * framerate;
		
		if (end_frame == 0)
		{
			end_frame = numberOfFrames - 1;
		}

		assert(start_frame < end_frame);
		assert(end_frame < numberOfFrames);

		vcap.set(CAP_PROP_POS_FRAMES, start_frame);
		int end = end_frame - start_frame;		
		while (frame_index <= end && vcap.read(current_frame))
		{
			//cout << frame_index << endl;
			track_frame(current_frame);
			
		}
	}
	if (params.visualization == 1)
	{
		cv::waitKey(0);
	}
}
