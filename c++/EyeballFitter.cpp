#include "EyeballFitter.h"
#include <opencv2/imgproc/types_c.h>

EyeballFitter::EyeballFitter()
{
	init();
}

EyeballFitter::~EyeballFitter()
{

}

void EyeballFitter::reset()
{
	frame_ = 0;
	eyePhi_ = vector<float>(2, 0);
	eyeTheta_ = vector<float>(2, 0);
	eyeRadius_ = 0.12;
	eyeIrisPhi_ = 0.42;
	eyeTrans_[0] = eyeTrans_backup_[0].clone();
	eyeTrans_[1] = eyeTrans_backup_[1].clone();
	eyeTrans_delta_backup_[0] = eyeTrans_backup_[0].clone();
	eyeTrans_delta_backup_[1] = eyeTrans_backup_[1].clone();
	landmark_ellipse_[0].reset();
	landmark_ellipse_[1].reset();
}

void EyeballFitter::loadCalibration(const string &filename)
{
	ifstream fin(filename);
	if (fin.is_open())
	{
		fin >> eyeRadius_;
		fin >> eyeIrisPhi_;
		fin >> eyeTrans_[0].at<float>(0, 0) >> eyeTrans_[0].at<float>(1, 0) >> eyeTrans_[0].at<float>(2, 0);
		fin >> eyeTrans_[1].at<float>(0, 0) >> eyeTrans_[1].at<float>(1, 0) >> eyeTrans_[1].at<float>(2, 0);
		fin.close();
	}
	else
		cout << "[WARNING] No Calibration File!" << endl;
}

void EyeballFitter::init()
{
	eye_lmk_id_.clear();
	eyeTrans_.clear();
	eyeTrans_backup_.clear();
	eyePhi_.clear();
	eyeTheta_.clear();
	irisLandMark_.clear();
	landmark_ellipse_.clear();

	eyeball_.initialize("data/eyeball_128_128.obj", "data/eye_iris_.png", "data/eye_iris_.png");

	// init for AABB setting
	vector<int> left_lmk_id = { 36, 37, 38, 39, 40, 41 };
	vector<int> right_lmk_id = { 42, 43, 44, 45, 46, 47 };
	eye_lmk_id_.push_back(left_lmk_id);
	eye_lmk_id_.push_back(right_lmk_id);

	// init params
	eyeRadius_ = 0.12;
	eyeIrisPhi_ = 0.42;
	
	cv::Mat left_trans(cv::Matx31f(-0.283510238, 0.471765947, 0.195450214));
	cv::Mat right_trans(cv::Matx31f(0.283510238, 0.471765947, 0.195450214));
	eyeTrans_.push_back(left_trans);
	eyeTrans_.push_back(right_trans);
	eyeTrans_backup_.push_back(left_trans);
	eyeTrans_backup_.push_back(right_trans);
	eyeTrans_delta_backup_.push_back(left_trans);
	eyeTrans_delta_backup_.push_back(right_trans);
	eyePhi_ = vector<float>(2, 0);
	eyeTheta_ = vector<float>(2, 0);

	// init ellipse
	landmark_ellipse_ = vector<MyEllipse>(2);

	// init frame
	frame_ = 0;
}

void EyeballFitter::setIrisLandmark(vector<cv::Mat> & lmk)
{
	irisLandMark_ = lmk;
}

void EyeballFitter::setEyeRadius(float radius)
{
	float diff = (0.12 - radius) * 0.2;
	eyeTrans_[0].at<float>(2, 0) += diff;
	eyeTrans_[1].at<float>(2, 0) += diff;
	eyeTrans_delta_backup_[0].at<float>(2, 0) += diff;
	eyeTrans_delta_backup_[1].at<float>(2, 0) += diff;
	eyeRadius_ = radius;
}

void EyeballFitter::setDeltaOnEyeTrans(vector<cv::Mat> &orbit_delta)
{
	for (int i = 0; i < orbit_delta.size(); i++)
	{
		eyeTrans_[i] = eyeTrans_backup_[i] + orbit_delta[i];
		eyeTrans_delta_backup_[i] = eyeTrans_[i].clone();
	}
}

cv::Mat EyeballFitter::phiTheta2Rotation(float phi, float theta)
{
	cv::Mat ret;
	cv::Mat matY(cv::Matx33f(
		cos(theta), 0, sin(theta),
		0, 1, 0,
		-sin(theta), 0, cos(theta)
	));
	cv::Mat matX(cv::Matx33f(
		1, 0, 0,
		0, cos(phi), -sin(phi),
		0, sin(phi), cos(phi)
	));
	ret = matY * matX;
	return ret;
}

cv::Mat EyeballFitter::yawPitchRoll2Rotation(float yaw, float pitch, float roll)
{
	cv::Mat ret;
	cv::Mat matZ(cv::Matx33f(
		cos(roll), -sin(roll), 0,
		sin(roll), cos(roll), 0,
		0, 0, 1
		));
	cv::Mat matY(cv::Matx33f(
		cos(pitch), 0, sin(pitch),
		0, 1, 0,
		-sin(pitch), 0, cos(pitch)
		));
	cv::Mat matX(cv::Matx33f(
		1, 0, 0,
		0, cos(yaw), -sin(yaw),
		0, sin(yaw), cos(yaw)
		));
	ret = matZ * matY * matX;
	return ret;
}

vector<float> EyeballFitter::rotation2YawPitchRoll(cv::Mat &rot)
{
	//float rx = atan2(rot.at<float>(1, 0), rot.at<float>(0, 0));
	//float ry = atan2(-rot.at<float>(2, 0), sqrt(rot.at<float>(2, 1) * rot.at<float>(2, 1) + rot.at<float>(2, 2) * rot.at<float>(2, 2)));
	//float rz = atan2(rot.at<float>(2, 1), rot.at<float>(2, 2));

	float rx, ry, rz;
	float sy = sqrt(rot.at<float>(0, 0) * rot.at<float>(0, 0) + rot.at<float>(1, 0) * rot.at<float>(1, 0));
	bool singular = sy < 1e-6; // If
	if (!singular)
	{
		rx = atan2(rot.at<float>(2, 1), rot.at<float>(2, 2));
		ry = atan2(-rot.at<float>(2, 0), sy);
		rz = atan2(rot.at<float>(1, 0), rot.at<float>(0, 0));
	}
	else
	{
		rx = atan2(-rot.at<float>(1, 2), rot.at<float>(1, 1));
		ry = atan2(-rot.at<float>(2, 0), sy);
		rz = 0;
	}

	vector<float> ret = { rx, ry, rz };
	return ret;
}

void EyeballFitter::getEyeAABB(vector<cv::Mat> &lmk2d)
{
	eyeAABB_.clear();
	eye_center_2d_.clear();
	for (int e = 0; e < EYE_NUM; e++)
	{
		cv::Mat center_2d = cv::Mat::zeros(2, 1, CV_32F);
		int top = INT_MAX; int left = INT_MAX;
		int bottom = 0; int right = 0;
		for (int l = 0; l < eye_lmk_id_[e].size(); l++)
		{
			int lmk_id = eye_lmk_id_[e][l];
			center_2d = center_2d + lmk2d[lmk_id];
			int x = int(lmk2d[lmk_id].at<float>(0, 0) + 0.5f);
			int y = int(lmk2d[lmk_id].at<float>(1, 0) + 0.5f);
			left = x < left ? x : left;
			top = y < top ? y : top;
			right = x > right ? x : right;
			bottom = y > bottom ? y : bottom;
		}
		center_2d = center_2d / eye_lmk_id_[e].size();
		eye_center_2d_.push_back(center_2d);
		eyeAABB_.push_back(cv::Rect(left, top, right - left, bottom - top));
	}
	eye_center_dis_ = sqrt((eye_center_2d_[0].at<float>(0, 0) - eye_center_2d_[1].at<float>(0, 0)) *
		(eye_center_2d_[0].at<float>(0, 0) - eye_center_2d_[1].at<float>(0, 0)) +
		(eye_center_2d_[0].at<float>(1, 0) - eye_center_2d_[1].at<float>(1, 0)) *
		(eye_center_2d_[0].at<float>(1, 0) - eye_center_2d_[1].at<float>(1, 0)));
	//cout << "Eye center distance: " << eye_center_dis_ << endl;
}

void EyeballFitter::estColor(cv::Mat &cv_img_gray)
{
	int colorHist[256] = { 0 };
	int total_pixel = 0;
	for (int e = 0; e < EYE_NUM; e++)
	{
		cv::Mat gray = cv_img_gray(eyeAABB_[e]);
		for (int r = 0; r < gray.rows; r++)
		{
			for (int c = 0; c < gray.cols; c++)
			{
				int color = (int)gray.at<uchar>(r, c);
				colorHist[color]++;
				total_pixel++;
			}
		}
	}

	/*float score[256] = { 0 };
	float maxScoreVal = 0;
	float minScore = std::numeric_limits<float>::max();
	int minID;
	int pixels = 0;
	float alpha = 450;
	for (int i = 255; i >= 0; i--)
	{
		pixels += colorHist[i];
		score[i] = i + alpha * pixels / total_pixel;
		if (score[i] < minScore)
		{
			minScore = score[i];
			minID = i;
		}
		if (colorHist[i] > maxScoreVal)
			maxScoreVal = colorHist[i];
	}
	maxColor_ = minID;

	float score1[256] = { 0 };
	minScore = std::numeric_limits<float>::max();
	int maxID;
	pixels = 0;
	for (int i = 0; i <= 255; i++)
	{
		pixels += colorHist[i];
		score1[i] = -i + alpha * pixels / total_pixel;
		if (score1[i] < minScore)
		{
			minScore = score1[i];
			maxID = i;
		}
	}
	minColor_ = maxID;*/

	float alpha = 20;
	float maxScore = 0, minScore = 0;
	int maxClr, minClr;
	for (int clr = 0; clr <= 255; clr++)
	{
		if (colorHist[clr] == 0)
			continue;
		float maxTemp = (float)clr / 255 + alpha * (float)colorHist[clr] / total_pixel;
		float minTemp = (1 - (float)clr / 255) + alpha * (float)colorHist[clr] / total_pixel;
		if (maxTemp > maxScore)
		{
			maxScore = maxTemp;
			maxClr = clr;
		}
		if (minTemp > minScore)
		{
			minScore = minTemp;
			minClr = clr;
		}
	}
	maxColor_ = maxClr;
	minColor_ = minClr;
}

void EyeballFitter::sampleEye()
{
	eyeSample_.clear();
	eyeColor_.clear();
	eyeIrisBoundary_.clear();

	float phi, theta;
	float deltaPhi = 0.02;
	float deltaTheta = M_PI * 2 / LONGI_NUM;
	int n = 2 * LATI_NUM - 1;

	for (int i = -n; i <= n; i += 2)
	{
		phi = eyeIrisPhi_ + i * deltaPhi;
		for (int j = 0; j < LONGI_NUM; j++)
		{
			cv::Mat sample(3, 1, CV_32F);
			theta = j * deltaTheta;
			sample.at<float>(0, 0) = eyeRadius_ * sin(phi) * cos(theta);
			sample.at<float>(1, 0) = eyeRadius_ * sin(phi) * sin(theta);
			sample.at<float>(2, 0) = eyeRadius_ * cos(phi);
			eyeSample_.push_back(sample);
			float clr = minColor_ + float(maxColor_ - minColor_) / (2 * n) * (i + n);
			eyeColor_.push_back(clr);
		}
	}
	//phi = M_PI / 2; 
	phi = eyeIrisPhi_;
	for (int j = 0; j < LONGI_NUM; j++)
	{
		cv::Mat sample(3, 1, CV_32F);
		theta = j * deltaTheta;
		sample.at<float>(0, 0) = eyeRadius_ * sin(phi) * cos(theta);
		sample.at<float>(1, 0) = eyeRadius_ * sin(phi) * sin(theta);
		sample.at<float>(2, 0) = eyeRadius_ * cos(phi);
		eyeIrisBoundary_.push_back(sample);
	}
}

void EyeballFitter::sampleEye2(float iris_phi)
{
	eyeSample_.clear();
	eyeIrisBoundary_.clear();

	float phi, theta;
	float deltaTheta = M_PI * 2 / LONGI_NUM2;

	phi = iris_phi;
	// left eye
	for (int j = 0; j < LONGI_NUM2; j++)
	{
		cv::Mat sample(3, 1, CV_32F);
		theta = M_PI * 2 - j * deltaTheta;
		sample.at<float>(0, 0) = eyeRadius_ * sin(phi) * cos(theta);
		sample.at<float>(1, 0) = eyeRadius_ * sin(phi) * sin(theta);
		sample.at<float>(2, 0) = eyeRadius_ * cos(phi);
		eyeSample_.push_back(sample);
	}
	// right eye
	for (int j = 0; j < LONGI_NUM2; j++)
	{
		cv::Mat sample(3, 1, CV_32F);
		theta = M_PI + j * deltaTheta;
		sample.at<float>(0, 0) = eyeRadius_ * sin(phi) * cos(theta);
		sample.at<float>(1, 0) = eyeRadius_ * sin(phi) * sin(theta);
		sample.at<float>(2, 0) = eyeRadius_ * cos(phi);
		eyeSample_.push_back(sample);
	}
}

float EyeballFitter::bilinear_interp(cv::Mat & M, float x, float y)
{
	int x_max = M.rows - 1; int y_max = M.cols - 1;
	float x_f = floor(x); float x_c = floor(x + 1);
	float y_f = floor(y); float y_c = floor(y + 1);

	int x_low = int(x_f + 0.5f); int x_high = int(x_c + 0.5f);
	int y_low = int(y_f + 0.5f); int y_high = int(y_c + 0.5f);

	float x1 = x - x_f; float x2 = x_c - x;
	float y1 = y - y_f; float y2 = y_c - y;

	float w1 = x2 * y2; float w2 = x1 * y2;
	float w3 = x2 * y1; float w4 = x1 * y1;

	float ret = w1 * M.at<uchar>(min(x_low, x_max), min(y_low, y_max)) +
		w2 * M.at<uchar>(min(x_high, x_max), min(y_low, y_max)) +
		w3 * M.at<uchar>(min(x_low, x_max), min(y_high, y_max)) +
		w4 * M.at<uchar>(min(x_high, x_max), min(y_high, y_max));
	return ret;
}

float EyeballFitter::getProjectColor(cv::Mat &gimg, cv::Mat &p3d, int f)
{
	int imgW = gimg.cols;
	int imgH = gimg.rows;
	cv::Mat proMat(cv::Matx33f(
		f, 0, float(imgW) / 2,
		0, f, float(imgH) / 2,
		0, 0, 1
		));
	cv::Mat point2D = proMat * p3d;
	float x = imgW - point2D.at<float>(0, 0) / p3d.at<float>(2, 0);
	float y = imgH - point2D.at<float>(1, 0) / p3d.at<float>(2, 0);
	float ret = bilinear_interp(gimg, y, x);
	return ret;
}

vector<cv::Mat> EyeballFitter::project3Dto2D(vector<cv::Mat> &points, int f, int imgW, int imgH)
{
	// construct project matrix
	cv::Mat proMat(cv::Matx33f(
		f, 0, float(imgW) / 2,
		0, f, float(imgH) / 2,
		0, 0, 1
		));
	vector<cv::Mat> ret;
	for (int p = 0; p < points.size(); p++)
	{
		cv::Mat p3d = points[p];
		cv::Mat p2d = proMat * p3d;
		cv::Mat temp(2, 1, CV_32F);
		temp.at<float>(0, 0) = imgW - p2d.at<float>(0, 0) / p3d.at<float>(2, 0);
		temp.at<float>(1, 0) = imgH - p2d.at<float>(1, 0) / p3d.at<float>(2, 0);
		ret.push_back(temp);
	}
	return ret;
}

void EyeballFitter::fit(cv::Mat &cv_img, vector<cv::Mat> &lmk2d, cv::Mat &rvec, cv::Mat &tvec, float f)
{
	face_Rvec_ = rvec;
	face_Tvec_ = tvec;
	camera_flength_ = f;
	getEyeAABB(lmk2d);
	cv::cvtColor(cv_img, gray_img_, CV_BGR2GRAY);
	estColor(gray_img_);
	sampleEye();

	EyeFitterPoseAdapter efpa(this);
	cv::Mat params(4, 1, CV_32F);
	params.at<float>(0, 0) = eyePhi_[0];
	params.at<float>(1, 0) = eyePhi_[1];
	params.at<float>(2, 0) = eyeTheta_[0];
	params.at<float>(3, 0) = eyeTheta_[1];
	cv::Mat outputs(eyeColor_);
	cv::vconcat(outputs, outputs, outputs);
	gns.setDerivStep(1e-3);
	gns.setMaxIter(3);
	gns.GaussNewton(efpa, params, outputs);
	eyePhi_[0] = params.at<float>(0, 0);
	eyePhi_[1] = params.at<float>(1, 0);
	eyeTheta_[0] = params.at<float>(2, 0);
	eyeTheta_[1] = params.at<float>(3, 0);
	//cout << params.t() << endl;
	//drawResult(cv_img);

	frame_++;
}

cv::Mat EyeballFitter::EyePoseObjectiveFunc(const cv::Mat &params)
{
	float phiL = params.at<float>(0, 0);
	float phiR = params.at<float>(1, 0);
	float thetaL = params.at<float>(2, 0);
	float thetaR = params.at<float>(3, 0);

	cv::Mat rmatL = phiTheta2Rotation(phiL, thetaL);
	cv::Mat rmatR = phiTheta2Rotation(phiR, thetaR);
	cv::Mat rmatF;
	cv::Rodrigues(face_Rvec_, rmatF);

	vector<cv::Mat> eyermat = { rmatL, rmatR };
	cv::Mat ret = cv::Mat::zeros(EYE_NUM * eyeSample_.size(), 1, CV_32F);
	for (int e = 0; e < EYE_NUM; e++)
	{
		for (int s = 0; s < eyeSample_.size(); s++)
		{
			cv::Mat temp = rmatF * (eyermat[e] * eyeSample_[s] + eyeTrans_[e]) + face_Tvec_;

			float clr = getProjectColor(gray_img_, temp, camera_flength_);
			ret.at<float>(e * eyeSample_.size() + s, 0) = clr;
		}
	}
	
	return ret;
}

void EyeballFitter::fit2(cv::Mat &cv_img, vector<cv::Mat> &lmk2d, cv::Mat &rvec, cv::Mat &tvec, float f, int status, string pre_path, int frame_id, int frame_type)
{
	frame_ = frame_id;
	face_Rvec_ = rvec;
	face_Tvec_ = tvec;
	camera_flength_ = f;
	img_width_ = cv_img.cols;
	img_height_ = cv_img.rows;
	status_ = status;
	setEyeImg(cv_img, lmk2d);

	if (frame_type == 3 && frame_id == 0)	// manual modify eyeball parameters
		loadCalibration("./debug/calibrateResult.txt");

	if (frame_type == 0)
	{
		//loadCalibration("./debug/calibrateResult.txt");
		return;
	}
	else if (frame_type == 1) // determine left eye position
	{
		int temp;
		//cout << "Frame " << frame_ << ", Solve Eye: (0: left, 1: right)" << endl;
		//cin >> temp;
		//fitIrisPhiAndEyeTrans(temp, 0, 0);
		cout << "Frame " << frame_id << ", Status: " << status << endl;
		fitIrisPhiAndEyeTrans(status, 0, 0);
		//drawResult2(cv_img);
		return;
	}
	else if (frame_type == 2)
	{
		int temp;
		//cout << "Frame " << frame_ << ", Solve Eye: (0: left, 1: right)" << endl;
		//cin >> temp;
		//fitIrisPhiAndEyeTrans(temp, 0, 0);
		cout << "Frame " << frame_id << ", Status: " << status << endl;
		fitIrisPhiAndEyeTrans(status, 0, 0);
		//eyeIrisPhi_ *= 0.9;
		//drawResult2(cv_img);
		return;
	}

	//drawResult2(cv_img);
	//save("./debug/eyeball/", "eye");
	sampleEye2(eyeIrisPhi_);
	EyeFitterPoseAdapter2 efpa2(this);
	cv::Mat params(4, 1, CV_32F);
	params.at<float>(0, 0) = eyePhi_[0];
	params.at<float>(1, 0) = eyePhi_[1];
	params.at<float>(2, 0) = eyeTheta_[0];
	params.at<float>(3, 0) = eyeTheta_[1];
	cv::Mat outputs;
	cv::vconcat(irisLandMark_, outputs);
	gns.setMaxIter(3);
	gns.GaussNewton(efpa2, params, outputs);

	eyePhi_[0] = 0;
	eyeTheta_[0] = 0;
	eyePhi_[1] = 0;
	eyeTheta_[1] = 0;
	//if (status == 0 || status == 2)
	//{
		eyePhi_[0] = params.at<float>(0, 0);
		eyeTheta_[0] = params.at<float>(2, 0);
	//}
	//if (status == 1 || status == 2)
	//{
		eyePhi_[1] = params.at<float>(1, 0);
		eyeTheta_[1] = params.at<float>(3, 0);
	//}

	//cout << params.t() << endl;
	//drawResult2(cv_img);
	if (frame_id == 0)
	{
		//drawResult2(cv_img);
		cout << "LP: " << eyePhi_[0] << endl
			 << "LT: " << eyeTheta_[0] << endl
			 << "RP: " << eyePhi_[1] << endl
			 << "RT: " << eyeTheta_[1] << endl;
	}
}

cv::Mat EyeballFitter::EyePoseObjectiveFunc2(const cv::Mat &params)
{
	float phiL = params.at<float>(0, 0);
	float phiR = params.at<float>(1, 0);
	float thetaL = params.at<float>(2, 0);
	float thetaR = params.at<float>(3, 0);

	cv::Mat rmatL = phiTheta2Rotation(phiL, thetaL);
	cv::Mat rmatR = phiTheta2Rotation(phiR, thetaR);
	cv::Mat rmatF;
	cv::Rodrigues(face_Rvec_, rmatF);

	vector<cv::Mat> eyermat = { rmatL, rmatR };
	
	int e;
	vector<cv::Mat> sample_transformed;
	for (int s = 0; s < eyeSample_.size(); s++)
	{
		if (s < eyeSample_.size() / EYE_NUM)
			e = 0;
		else
			e = 1;
		cv::Mat temp = rmatF * (eyermat[e] * eyeSample_[s] + eyeTrans_[e]) + face_Tvec_;
		sample_transformed.push_back(temp);
	}
	vector<cv::Mat> proj_key_vertices = project3Dto2D(sample_transformed, camera_flength_, img_width_, img_height_);
	cv::Mat ret;
	cv::vconcat(proj_key_vertices, ret);
	return ret;
}

void EyeballFitter::drawResult(cv::Mat &cv_img)
{
	int imgW = cv_img.cols;
	int imgH = cv_img.rows;
	cv::Mat rmatF;
	cv::Rodrigues(face_Rvec_, rmatF);
	vector<cv::Mat> boundary;
	for (int e = 0; e < EYE_NUM; e++)
	{	// for method based on color
		cv::Mat rmatE = phiTheta2Rotation(eyePhi_[e], eyeTheta_[e]);
		for (int b = 0; b < eyeIrisBoundary_.size(); b++)
		{
			cv::Mat temp = rmatF * (rmatE * eyeIrisBoundary_[b] + eyeTrans_[e]) + face_Tvec_;
			boundary.push_back(temp);
		}
	}
	vector<cv::Mat> proj_key_vertices = project3Dto2D(boundary, camera_flength_, imgW, imgH);
	for (int i = 0; i < proj_key_vertices.size(); i++)
	{
		int x = int(proj_key_vertices[i].at<float>(0, 0) + 0.5f);
		int y = int(proj_key_vertices[i].at<float>(1, 0) + 0.5f);
		cv::circle(cv_img, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);
	}

	cv::imshow("Result", cv_img);
	cv::waitKey(30);
	cv::imwrite("result.png", cv_img);

	float pos[6] = {
		eyeTrans_[0].at<float>(0, 0),
		eyeTrans_[0].at<float>(1, 0),
		eyeTrans_[0].at<float>(2, 0),
		eyeTrans_[1].at<float>(0, 0),
		eyeTrans_[1].at<float>(1, 0),
		eyeTrans_[1].at<float>(2, 0)
	};
	THU::Matrix3f rot;
	THU::Vector3f tran;
	cv::cv2eigen(rmatF, rot);
	cv::cv2eigen(face_Tvec_, tran);
	eyeball_.updateEyeballSize(eyeRadius_);
	eyeball_.updateTextureCordinates(eyeIrisPhi_);
	eyeball_.setEyeballPos(pos);
	//eyeball_.outputMesh(1.0, rot, tran, "eye");
	eyeball_.outputColorMesh(1.0, eyePhi_, eyeTheta_, rot, tran, "eye");
}

void EyeballFitter::drawResult2(cv::Mat &cv_img)
{
	int imgW = cv_img.cols;
	int imgH = cv_img.rows;
	cv::Mat show = cv_img.clone();
	cv::Mat rmatF;
	cv::Rodrigues(face_Rvec_, rmatF);
	vector<cv::Mat> boundary;
	for (int b = 0; b < eyeSample_.size(); b++)
	{	// for method based on iris landmark
		int e;
		if (b < eyeSample_.size() / EYE_NUM)
			e = 0;
		else
			e = 1;
		cv::Mat rmatE = phiTheta2Rotation(eyePhi_[e], eyeTheta_[e]);
		cv::Mat temp = rmatF * (rmatE * eyeSample_[b] + eyeTrans_[e]) + face_Tvec_;
		boundary.push_back(temp);
	}
	// draw proj
	vector<cv::Mat> proj_key_vertices = project3Dto2D(boundary, camera_flength_, imgW, imgH);
	for (int i = 0; i < proj_key_vertices.size(); i++)
	{
		int x = int(proj_key_vertices[i].at<float>(0, 0) + 0.5f);
		int y = int(proj_key_vertices[i].at<float>(1, 0) + 0.5f);
		cv::circle(show, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);
	}

	
	//cv::imshow("Result", cv_img);
	//cv::waitKey(30);
	cv::imwrite("./debug/eyeball/eye_" + to_string(frame_) + ".png", show);

	// save 2d error
	/*ofstream fout("./debug/eyeball/iriserror_" + to_string(frame_) + ".txt");
	for (int i = 0; i < proj_key_vertices.size(); i++)
	{
		float err = sqrt((irisLandMark_[i].at<float>(0, 0) - proj_key_vertices[i].at<float>(0, 0)) *
			(irisLandMark_[i].at<float>(0, 0) - proj_key_vertices[i].at<float>(0, 0)) +
			(irisLandMark_[i].at<float>(1, 0) - proj_key_vertices[i].at<float>(1, 0)) *
			(irisLandMark_[i].at<float>(1, 0) - proj_key_vertices[i].at<float>(1, 0)));
		fout << err / eye_center_dis_ << endl;
	}
	fout << eye_center_dis_ << endl;
	fout.close();*/
}

void EyeballFitter::save(const string & out_dir, const string & filename)
{
	cv::Mat rmatF;
	cv::Rodrigues(face_Rvec_, rmatF);
	float pos[6] = {
		eyeTrans_[0].at<float>(0, 0),
		eyeTrans_[0].at<float>(1, 0),
		eyeTrans_[0].at<float>(2, 0),
		eyeTrans_[1].at<float>(0, 0),
		eyeTrans_[1].at<float>(1, 0),
		eyeTrans_[1].at<float>(2, 0)
	};
	THU::Matrix3f rot;
	THU::Vector3f tran;
	cv::cv2eigen(rmatF, rot);
	cv::cv2eigen(face_Tvec_, tran);
	eyeball_.updateEyeballSize(eyeRadius_);
	eyeball_.updateTextureCordinates(eyeIrisPhi_);
	eyeball_.setEyeballPos(pos);
	eyeball_.outputMesh(1.0, rot, tran, "eye");
	eyeball_.outputColorMesh(1.0, eyePhi_, eyeTheta_, rot, tran, out_dir + "/" + filename + "_eyeball");
}

void EyeballFitter::saveJson(Json::Value & item)
{
	if (frame_ <= 0)
	{
		// eye radius
		item["eye_radius"] = eyeRadius_;
		// iris radius
		item["iris_radius"] = eyeIrisPhi_;
		// left eye
		item["left_eye_trans"].append(eyeTrans_[0].at<float>(0, 0));
		item["left_eye_trans"].append(eyeTrans_[0].at<float>(1, 0));
		item["left_eye_trans"].append(eyeTrans_[0].at<float>(2, 0));
		// right eye
		item["right_eye_trans"].append(eyeTrans_[1].at<float>(0, 0));
		item["right_eye_trans"].append(eyeTrans_[1].at<float>(1, 0));
		item["right_eye_trans"].append(eyeTrans_[1].at<float>(2, 0));
	}

	item["left_phi"] = eyePhi_[0];
	item["left_theta"] = eyeTheta_[0];
	item["right_phi"] = eyePhi_[1];
	item["right_theta"] = eyeTheta_[1];

	/*int imgW = face_img_.cols;
	int imgH = face_img_.rows;
	cv::Mat rmatF;
	cv::Rodrigues(face_Rvec_, rmatF);
	vector<cv::Mat> boundary;
	for (int b = 0; b < eyeSample_.size(); b++)
	{	// for method based on iris landmark
		int e;
		if (b < eyeSample_.size() / EYE_NUM)
			e = 0;
		else
			e = 1;
		cv::Mat rmatE = phiTheta2Rotation(eyePhi_[e], eyeTheta_[e]);
		cv::Mat temp = rmatF * (rmatE * eyeSample_[b] + eyeTrans_[e]) + face_Tvec_;
		boundary.push_back(temp);
	}

	vector<cv::Mat> proj_key_vertices = project3Dto2D(boundary, camera_flength_, imgW, imgH);

	// save 2d error
	for (int i = 0; i < proj_key_vertices.size(); i++)
	{
		float err = sqrt((irisLandMark_[i].at<float>(0, 0) - proj_key_vertices[i].at<float>(0, 0)) *
			(irisLandMark_[i].at<float>(0, 0) - proj_key_vertices[i].at<float>(0, 0)) +
			(irisLandMark_[i].at<float>(1, 0) - proj_key_vertices[i].at<float>(1, 0)) *
			(irisLandMark_[i].at<float>(1, 0) - proj_key_vertices[i].at<float>(1, 0)));
		if ((status_ == 0 || status_ == 2) && i < proj_key_vertices.size() / EYE_NUM)
			item["eyeball_err"].append(err / eye_center_dis_);
		else if ((status_ == 1 || status_ == 2) && i >= proj_key_vertices.size() / EYE_NUM)
			item["eyeball_err"].append(err / eye_center_dis_);
		else
			item["eyeball_err"].append(0.0f);
	}*/
}

// Ellipse Added
cv::Rect EyeballFitter::getCropRect(cv::Rect &rec)
{
	int left = rec.x; int right = left + rec.width;
	int top = rec.y; int bottom = top + rec.height;
	int centerX = (left + right) / 2;
	int centerY = (top + bottom) / 2;
	int h = rec.height;
	int w = rec.width;
	float extentionScale = 1.8;
	int scaleBase = h / 2.0 > w / 3.0 ? h * extentionScale / 2 : w * extentionScale / 3;
	top = centerY - scaleBase;
	left = centerX - 1.5 * scaleBase;
	h = 2 * scaleBase;
	w = 3 * scaleBase;

	cv::Rect cropRect(left, top, w, h);
	return cropRect;
}

void EyeballFitter::setEyeImg(cv::Mat &cv_img, vector<cv::Mat> &lmk2d)
{
	eye_img_.clear();
	eyeCrop_.clear();
	getEyeAABB(lmk2d);

	cv::Rect crop_L = getCropRect(eyeAABB_[0]);
	cv::Rect crop_R = getCropRect(eyeAABB_[1]);

	eye_img_.push_back(cv_img(crop_L));
	eye_img_.push_back(cv_img(crop_R));
	eyeCrop_.push_back(crop_L);
	eyeCrop_.push_back(crop_R);
	face_img_ = cv_img.clone();
}

void EyeballFitter::singleCalibrate(cv::Mat &cv_img, vector<cv::Mat> &lmk2d, cv::Mat &rvec, cv::Mat &tvec, int f)
{
	face_Rvec_ = rvec;
	face_Tvec_ = tvec;
	camera_flength_ = f;
	img_width_ = cv_img.cols;
	img_height_ = cv_img.rows;

	setEyeImg(cv_img, lmk2d);
	sampleEye2(eyeIrisPhi_);
	fitEllipse();
}

void EyeballFitter::fitEllipse()
{
	eye_center_2d_normed_.clear();
	for (int eye = 0; eye < EYE_NUM; eye++)
	{
		landmark_ellipse_[eye].reset();
		cv::Mat offset(cv::Matx21f(eyeCrop_[eye].x, eyeCrop_[eye].y));
		int imgW = eye_img_[eye].cols;
		// get eye center
		cv::Mat rmatAnti = cv::Mat(cv::Matx33f(
			-1, 0, 0,
			0, 1, 0,
			0, 0, -1
			));
		cv::Mat rmat;
		cv::Rodrigues(face_Rvec_, rmat);
		cv::Mat temp = rmatAnti * rmat;
		vector<float> faceypr = rotation2YawPitchRoll(temp);
		cout << "Face Rotation: " << faceypr[0] << " " << faceypr[1] << " " << faceypr[2] << endl;
		/*cv::Mat center_3d = rmat * eyeTrans_[eye] + face_Tvec_;
		vector<cv::Mat> temp_3d = { center_3d };
		vector<cv::Mat> temp_2d = project3Dto2D(temp_3d, camera_flength_, img_width_, img_height_);
		cv::Mat center_2d_normed = (temp_2d[0] - offset) / imgW;
		eye_center_2d_normed_.push_back(center_2d_normed);*/
		cv::Mat center_2d_normed = (eye_center_2d_[eye] - offset) / imgW;
		eye_center_2d_normed_.push_back(center_2d_normed);
		// normalize by crop image width
		vector<cv::Mat> iris_lmk_norm;
		int landmark_half_num = irisLandMark_.size() / 2;
		for (int i = eye * landmark_half_num; i < (eye + 1) * landmark_half_num; i++)
		{
			cv::Mat temp = (irisLandMark_[i] - offset) / imgW - center_2d_normed;
			iris_lmk_norm.push_back(temp);
		}
		landmark_ellipse_[eye].fit(iris_lmk_norm);
		landmark_ellipse_[eye].draw(eye_img_[eye], center_2d_normed, eye); // show ellipse fitting result

		// init eye rotation with ellipse parameters
		vector<float> ellipse_params = landmark_ellipse_[eye].getABT(); // [a/b, theta, x_off, y_off]
		float circle_phi1 = acos(ellipse_params[0]);
		float circle_phi2 = -acos(ellipse_params[0]);
		float circle_the = -ellipse_params[1];

		cv::Mat vec = cv::Mat(cv::Matx31f(-sin(circle_the), cos(circle_the), 0));
		cv::Mat rvec1 = vec * circle_phi1;
		cv::Mat rvec2 = vec * circle_phi2;
		cv::Mat total_rmat1, total_rmat2;
		cv::Rodrigues(rvec1, total_rmat1);
		cv::Rodrigues(rvec2, total_rmat2);
		
		vector<float> totalxyz1 = rotation2YawPitchRoll(total_rmat1);
		cout << "Total Rotation 1: " << totalxyz1[0] << " " << totalxyz1[1] << " " << totalxyz1[2] << endl;
		vector<float> totalxyz2 = rotation2YawPitchRoll(total_rmat2);
		cout << "Total Rotation 2: " << totalxyz2[0] << " " << totalxyz2[1] << " " << totalxyz2[2] << endl;
		
		cv::Mat rmat_total1 = rmatAnti * total_rmat1;
		cv::Mat rmat_total2 = rmatAnti * total_rmat2;
		cv::Mat rmat_eye1 = rmat.inv() * rmat_total1;
		cv::Mat rmat_eye2 = rmat.inv() * rmat_total2;
		vector<float> angles1 = rotation2YawPitchRoll(rmat_eye1);
		vector<float> angles2 = rotation2YawPitchRoll(rmat_eye2);
		//float signPhi = angles[0] * ellipse_params[3];
		//float signTheta = angles[1] * ellipse_params[2];
		//eyePhi_[eye] = signPhi > 0 ? angles[0] : -angles[0]; // make sure eye_phi and y_off both >0 or both <0 ��HAVE PROBLEM��
		//eyeTheta_[eye] = signTheta > 0 ? angles[1] : -angles[1]; // make sure eye_theta and x_off both >0 or both <0
		//cv::Mat valid = phiTheta2Rotation(eyePhi_[eye], eyeTheta_[eye]);
		cout << rmat_eye1 << endl;
		cout << "Eyeball Rotation 1: " << angles1[0] << " " << angles1[1] << " " << angles1[2] << endl; // the last one should be close to 0.
		cout << rmat_eye2 << endl;
		cout << "Eyeball Rotation 2: " << angles2[0] << " " << angles2[1] << " " << angles2[2] << endl; // the last one should be close to 0.
		//cout << valid << endl;

		// auto choose
		//cout << eyeTrans_delta_backup_[eye].t() << endl;
		// suppose 0
		eyeIrisPhi_ = 0.42;
		eyeTrans_[eye] = eyeTrans_delta_backup_[eye].clone();
		fitIrisPhiAndEyeTrans(eye, angles1[0], angles1[1]);
		cv::Mat trans_1 = eyeTrans_[eye].clone();
		// suppose 1
		eyeIrisPhi_ = 0.42;
		eyeTrans_[eye] = eyeTrans_delta_backup_[eye].clone();
		fitIrisPhiAndEyeTrans(eye, angles2[0], angles2[1]);
		cv::Mat trans_2 = eyeTrans_[eye].clone();
		// choose smaller diff
		float diff_1 = cv::norm(trans_1, eyeTrans_delta_backup_[eye], cv::NORM_L2);
		float diff_2 = cv::norm(trans_2, eyeTrans_delta_backup_[eye], cv::NORM_L2);
		//cout << diff_1 << " " << diff_2 << endl;
		if (diff_1 < diff_2)
		{
			eyePhi_[eye] = angles1[0];
			eyeTheta_[eye] = angles1[1];
		}
		else
		{
			eyePhi_[eye] = angles2[0];
			eyeTheta_[eye] = angles2[1];
		}

		// manual choose
		/*int id;
		cout << "Choose (0/1): ";
		cin >> id;
		eyePhi_[eye] = id == 0 ? angles1[0] : angles2[0];
		eyeTheta_[eye] = id == 0 ? angles1[1] : angles2[1];*/


		cout << eyePhi_[eye] << " " << eyeTheta_[eye] << endl;
		cout << endl;
	}
}


void EyeballFitter::fitIrisPhiAndEyeTrans(int eye, float phi, float theta)
{
	eye_mode_ = eye;
	eyePhi_[eye] = phi;
	eyeTheta_[eye] = theta;

	IrisPhiEyeTransAdapter ipeta(this);

	cv::Mat params(3, 1, CV_32F);
	params.at<float>(0, 0) = eyeIrisPhi_;
	params.at<float>(1, 0) = eyeTrans_[eye].at<float>(0, 0);
	params.at<float>(2, 0) = eyeTrans_[eye].at<float>(1, 0);

	cv::Mat outputs;
	cv::vconcat(irisLandMark_, outputs);
	if (eye == 0)
		outputs = outputs.rowRange(0, outputs.rows / EYE_NUM);
	else
		outputs = outputs.rowRange(outputs.rows / EYE_NUM, outputs.rows);
	gns.setMaxIter(3);
	gns.setDerivStep(1e-3);
	gns.GaussNewton(ipeta, params, outputs);

	eyeIrisPhi_ = params.at<float>(0, 0);
	eyeTrans_[eye].at<float>(0, 0) = params.at<float>(1, 0);
	eyeTrans_[eye].at<float>(1, 0) = params.at<float>(2, 0);
	cout << "Iris Phi: " << eyeIrisPhi_ << endl;
	cout << "Eye Position: " << eyeTrans_[eye].t() << " " << eyeTrans_[eye].t() << endl;
}

cv::Mat EyeballFitter::IrisPhiEyeTransObj(const cv::Mat &params)
{
	float phi = eyePhi_[eye_mode_];
	float theta = eyeTheta_[eye_mode_];

	cv::Mat rmat = phiTheta2Rotation(phi, theta);
	cv::Mat rmatF;
	cv::Rodrigues(face_Rvec_, rmatF);

	sampleEye2(params.at<float>(0, 0)); // iris phi

	cv::Mat trans(3, 1, CV_32F);
	trans.at<float>(0, 0) = params.at<float>(1, 0);
	trans.at<float>(1, 0) = params.at<float>(2, 0);
	trans.at<float>(2, 0) = eyeTrans_[eye_mode_].at<float>(2, 0);

	vector<cv::Mat> sample_transformed;
	int start = eye_mode_ == 0 ? 0 : eyeSample_.size() / EYE_NUM;
	int end = eye_mode_ == 0 ? eyeSample_.size() / EYE_NUM : eyeSample_.size();
	for (int s = start; s < end; s++)
	{
		cv::Mat temp = rmatF * (rmat * eyeSample_[s] + trans) + face_Tvec_;
		sample_transformed.push_back(temp);
	}
	vector<cv::Mat> proj_key_vertices = project3Dto2D(sample_transformed, camera_flength_, img_width_, img_height_);
	cv::Mat ret;
	cv::vconcat(proj_key_vertices, ret);
	return ret;
}

void EyeballFitter::setFittingResult(Json::Value& item)
{
	// first frame
	if (frame_ <= 2)
	{
		if (!item["left_eye_trans"].empty())
		{
			cv::Mat temp(3, 1, CV_32F);
			temp.at<float>(0, 0) = item["left_eye_trans"][0].asFloat();
			temp.at<float>(1, 0) = item["left_eye_trans"][1].asFloat();
			temp.at<float>(2, 0) = item["left_eye_trans"][2].asFloat();
			eyeTrans_[0] = temp;
		}
		if (!item["right_eye_trans"].empty())
		{
			cv::Mat temp(3, 1, CV_32F);
			temp.at<float>(0, 0) = item["right_eye_trans"][0].asFloat();
			temp.at<float>(1, 0) = item["right_eye_trans"][1].asFloat();
			temp.at<float>(2, 0) = item["right_eye_trans"][2].asFloat();
			eyeTrans_[1] = temp;
		}
		if (!item["iris_radius"].empty())
		{
			eyeIrisPhi_ = item["iris_radius"].asFloat();
		}
		if (!item["eye_radius"].empty())
		{
			eyeRadius_ = item["eye_radius"].asFloat();
		}
	}

	// eyeball rotation
	float phi_0, phi_1, the_0, the_1;
	phi_0 = item["left_phi"].asFloat();
	phi_1 = item["right_phi"].asFloat();
	the_0 = item["left_theta"].asFloat();
	the_1 = item["right_theta"].asFloat();
	setEyePhi(phi_0, phi_1);
	setEyeTheta(the_0, the_1);
	// face rotation and translation
	cv::Mat rvec(3, 1, CV_32F);
	cv::Mat tvec(3, 1, CV_32F);
	rvec.at<float>(0, 0) = item["rvec"][0].asFloat();
	rvec.at<float>(1, 0) = item["rvec"][1].asFloat();
	rvec.at<float>(2, 0) = item["rvec"][2].asFloat();
	tvec.at<float>(0, 0) = item["tvec"][0].asFloat();
	tvec.at<float>(1, 0) = item["tvec"][1].asFloat();
	tvec.at<float>(2, 0) = item["tvec"][2].asFloat();
	face_Rvec_ = rvec.clone();
	face_Tvec_ = tvec.clone();

	frame_++;
}

int main() {
    EyeballFitter fitter;
    fitter.eye_lmk_id_ = {{0, 1, 2, 3}, {4, 5, 6, 7}};

    vector<cv::Mat> lmk2d = {
        (cv::Mat_<float>(2, 1) << 10, 20),
        (cv::Mat_<float>(2, 1) << 15, 25),
        (cv::Mat_<float>(2, 1) << 30, 40),
        (cv::Mat_<float>(2, 1) << 35, 45),
        (cv::Mat_<float>(2, 1) << 50, 60),
        (cv::Mat_<float>(2, 1) << 55, 65),
        (cv::Mat_<float>(2, 1) << 70, 80),
        (cv::Mat_<float>(2, 1) << 75, 85)
    };

    fitter.getEyeAABB(lmk2d);

    for (const auto &rect : fitter.eyeAABB_) {
        cout << "AABB: " << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << endl;
    }
    cout << "Eye center distance: " << fitter.eye_center_dis_ << endl;

    return 0;
}