#ifndef EYEBALLMODEL_H
#define EYEBALLMODEL_H

#include <THUtils/Mesh/trimesh.h>
#include <THUtils/Mesh/trimesh.h>
#include <opencv2/core.hpp>

class EyeballModel
{
	typedef struct _TextureUnit
	{
		int texImgW;
		int texImgH;
		int texImgBPP;
		uchar* texImgData;
		_TextureUnit() : texImgH(0), texImgW(0), texImgBPP(0), texImgData(nullptr) {}
	} TextureUnit;

public:
	EyeballModel();
	~EyeballModel();

	void initialize(const std::string& fileName, const std::string& irisTexName, const std::string& corneaTexName);
	void setEyeballPos(const float* ep);
	void setPupilPos(const float* pp);

	const std::vector<THU::Vector3f>& eyePositions() { return eyePos; }
	const std::vector<THU::Vector3f>& pupilPositions() { return pupilPos; }

	const THU::TriMesh* getMesh() { return eyeballMesh; }
	const THU::TriMesh* getSphere() { return sphere; }
	const int getIrisVertexNum() { return irisVertexNum; }
	const int getIrisFaceNum() { return irisFaceNum; }
	const int getCorneaFaceNum() { return eyeballMesh->faceNum() - irisFaceNum; }
	const TextureUnit* getIrisTexture() { return irisTex; }
	const TextureUnit* getCorneaTexture() { return corneaTex; }

	void updateTextureCordinates(float irisP);
	void updateEyeballSize(float radius);

	void outputMesh(const float scale, const THU::Matrix3f& rot, const THU::Vector3f& trans, const std::string& label = "");
	void outputColorMesh(const float scale, std::vector<float> &phi, std::vector<float> &theta, const THU::Matrix3f& rot, const THU::Vector3f& trans, const std::string& label = "");

	bool useTexture;

private:
	void loadObject(const std::string& fileName);
	void loadTexture(TextureUnit*& texUnit, const std::string& texImgName);
	THU::Matrix3f phiTheta2Rotation(float phi, float theta);

	THU::TriMesh* eyeballMesh;
	THU::TriMesh* sphere;
	std::vector<THU::Vector3f> eyePos;
	std::vector<THU::Vector3f> pupilPos;
	float irisPhi;

	int irisVertexNum;
	int irisFaceNum;

	TextureUnit* irisTex;
	TextureUnit* corneaTex;
};

#endif