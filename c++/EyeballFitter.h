#pragma once
#include "GaussNewtonSolver.h"
#include "EyeballModel.h"
#include "MyEllipse.h"
#include <THUtils/Tools/timer.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <json/json.h>

using namespace std;

#define EYE_NUM 2
#define LONGI_NUM 32
#define LATI_NUM 2
#define LONGI_NUM2 19

class EyeballFitter
{
public:
	EyeballFitter();
	~EyeballFitter();
	void fit(cv::Mat &cv_img, vector<cv::Mat> &lmk2d, cv::Mat &rvec, cv::Mat &tvec, float f); // base on color
	//void fit2(cv::Mat &cv_img, vector<cv::Mat> &lmk2d, cv::Mat &rvec, cv::Mat &tvec, float f, int status, string pre_path); // base on iris landmarks
	void fit2(cv::Mat& cv_img, vector<cv::Mat>& lmk2d, cv::Mat& rvec, cv::Mat& tvec, float f, int status, string pre_path, int frame_id, int frame_type); // base on iris landmarks
	cv::Mat EyePoseObjectiveFunc(const cv::Mat &params);  // base on color
	cv::Mat EyePoseObjectiveFunc2(const cv::Mat &params); // base on iris landmarks
	void setDeltaOnEyeTrans(vector<cv::Mat> &orbit_delta);

	void singleCalibrate(cv::Mat &cv_img, vector<cv::Mat> &lmk2d, cv::Mat &rvec, cv::Mat &tvec, int f);
	void fitEllipse();
	void fitIrisPhiAndEyeTrans(int eye, float phi, float theta);
	cv::Mat IrisPhiEyeTransObj(const cv::Mat &params);

	void setEyePhi(float phi_0, float phi_1) { eyePhi_[0] = phi_0; eyePhi_[1] = phi_1; };
	void setEyeTheta(float the_0, float the_1) { eyeTheta_[0] = the_0; eyeTheta_[1] = the_1; };
	void setFittingResult(Json::Value& item);

	void setIrisLandmark(vector<cv::Mat> & lmk);
	void setEyeRadius(float radius);
	void reset();
	void skip() { frame_++; status_ = -1; };
	void save(const string & out_dir, const string & filename);
	void saveJson(Json::Value & item);

	vector<float> getEyePhi() { return eyePhi_; };
	vector<float> getEyeTheta() { return eyeTheta_; };
	float getEyeRadius() { return eyeRadius_; };
	float getEyeIrisPhi() { return eyeIrisPhi_; };
	vector<cv::Mat> getEyeTrans() { return eyeTrans_; };
	vector<cv::Mat> getEyeTransDelta() { return eyeTrans_delta_backup_; };
	vector<cv::Mat> getIrisLandMark() { return irisLandMark_; };

private:
	void init();
	void loadCalibration(const string &filename);
	cv::Mat phiTheta2Rotation(float phi, float theta);
	cv::Mat yawPitchRoll2Rotation(float yaw, float pitch, float roll);
	vector<float> rotation2YawPitchRoll(cv::Mat &rot);
	float getProjectColor(cv::Mat &gimg, cv::Mat &p3d, int f);
	float bilinear_interp(cv::Mat & M, float x, float y);
	vector<cv::Mat> project3Dto2D(vector<cv::Mat> &points, int f, int imgW, int imgH);
	void getEyeAABB(vector<cv::Mat> &lmk2d);
	void estColor(cv::Mat &cv_img_gray);
	void sampleEye(); // samples for color fitting
	void sampleEye2(float iris_phi); // samples for Iris fitting 
	void drawResult(cv::Mat &cv_img);
	void drawResult2(cv::Mat &cv_img);

	cv::Rect getCropRect(cv::Rect &rec); // add for ellipse
	void setEyeImg(cv::Mat &cv_img, vector<cv::Mat> &lmk2d); // add for ellipse
	cv::Mat face_img_;	// add for ellipse
	vector<cv::Mat> eye_img_; // add for ellipse
	vector<cv::Rect> eyeCrop_; // add for ellipse
	vector<cv::Mat> eye_center_2d_; // add for ellipse
	vector<cv::Mat> eye_center_2d_normed_; // add for ellipse

	// "left" and "right" are in the image space
	vector<vector<int>> eye_lmk_id_; // [2 x 6] array
	vector<cv::Rect> eyeAABB_;
	vector<cv::Mat> eyeSample_;
	vector<cv::Mat> irisLandMark_; // iris detection result [38, 2]
	vector<cv::Mat> eyeIrisBoundary_;
	vector<float> eyeColor_;
	EyeballModel eyeball_;
	int maxColor_;
	int minColor_;
	cv::Mat gray_img_;

	GaussNewtonSolver gns;
	vector<MyEllipse> landmark_ellipse_;

	float eyeRadius_;
	float eyeIrisPhi_;
	vector<cv::Mat> eyeTrans_backup_;
	vector<cv::Mat> eyeTrans_;
	vector<float> eyePhi_; // [2 x 1] array up-down
	vector<float> eyeTheta_; // [2 x 1] array left-right
	cv::Mat face_Rvec_;
	cv::Mat face_Tvec_;
	int camera_flength_;
	int img_width_, img_height_;

	int frame_;
	float eye_center_dis_;
	int eye_mode_; // the eye calibrating 0: left eye; 1: right eye;
	vector<cv::Mat> eyeTrans_delta_backup_; // backup eyeTrans + orbit_delta
	int status_;
};

class EyeFitterPoseAdapter{
public:
	EyeFitterPoseAdapter(EyeballFitter *rs_) :rs(rs_) {};
	cv::Mat operator() (const cv::Mat &params) { return rs->EyePoseObjectiveFunc(params); };
private:
	EyeballFitter *rs;
};

class EyeFitterPoseAdapter2{
public:
	EyeFitterPoseAdapter2(EyeballFitter *rs_) :rs(rs_) {};
	cv::Mat operator() (const cv::Mat &params) { return rs->EyePoseObjectiveFunc2(params); };
private:
	EyeballFitter *rs;
};

class IrisPhiEyeTransAdapter{
public:
	IrisPhiEyeTransAdapter(EyeballFitter *rs_) :rs(rs_) {};
	cv::Mat operator() (const cv::Mat &params) { return rs->IrisPhiEyeTransObj(params); };
private:
	EyeballFitter *rs;
};