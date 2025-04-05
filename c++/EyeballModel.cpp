#include"EyeballModel.h"

namespace Gdiplus
{
	using std::min;
	using std::max;
};
#include "atlimage.h"
#include <fstream>

#include <THUtils/Tools/memutils.h>
#include <THUtils/IO/logutils.h>

EyeballModel::EyeballModel() :
	eyeballMesh(nullptr),
	sphere(nullptr),
	irisVertexNum(/*81*/0),
	irisFaceNum(/*144*/0),
	irisTex(nullptr),
	corneaTex(nullptr),
	useTexture(false),
	irisPhi(/*0.471*/0.25 * M_PI_2)
{
	eyePos.insert(eyePos.begin(), 2, THU::Vector3f::Zero());
	pupilPos.insert(pupilPos.begin(), 2, THU::Vector3f::Zero());
}

EyeballModel::~EyeballModel()
{
	if (eyeballMesh != nullptr)
		THU::safeDelete(eyeballMesh);

	eyePos.clear();
	pupilPos.clear();
}

void EyeballModel::initialize(const std::string& fileName, const std::string& irisTexName, const std::string& corneaTexName)
{
	loadObject(fileName);
	loadTexture(irisTex, irisTexName);
	loadTexture(corneaTex, corneaTexName);

	if (irisTex != nullptr && corneaTex != nullptr)
		useTexture = true;
}

void EyeballModel::loadObject(const std::string& fileName)
{
	if (fileName.empty())
	{
		PRINT_ERROR("Lode eyeball object failed.");
		return;
	}

	if (eyeballMesh != nullptr)
		THU::safeDelete(eyeballMesh);

	eyeballMesh = new THU::TriMesh;
	eyeballMesh->loadMesh(fileName);
	eyeballMesh->initTopology();
	//updateTextureCordinates(0.34);

	//if (sphere != nullptr)
	//	THU::safeDelete(sphere);

	//sphere = new THU::TriMesh;
	//sphere->loadMesh("data/sphere.obj");
	//sphere->loadMesh("data/sphere_s.obj");
}

void EyeballModel::loadTexture(TextureUnit*& textUnit, const std::string& texImgName)
{
	if (texImgName.empty())
	{
		PRINT_INFO("No texture for eyeball.");
		return;
	}

	CImage img;
	CString texName(texImgName.c_str());
	if (FAILED(img.Load(texName)))
	{
		PRINT_WARNING("Cannot load image " << texImgName);
		return;
	}

	if (textUnit != nullptr)
		THU::safeDelete(textUnit);
	textUnit = new TextureUnit;

	textUnit->texImgW = img.GetWidth();
	textUnit->texImgH = img.GetHeight();
	int pitch = img.GetPitch();
	void* pImgData = img.GetBits();
	if (pitch < 0)
		pImgData = img.GetPixelAddress(0, textUnit->texImgH - 1);
	else
		pImgData = img.GetPixelAddress(0, 0);
	textUnit->texImgBPP = img.GetBPP();
	THU::safeDeleteArray(textUnit->texImgData);
	int dataSize = textUnit->texImgH * textUnit->texImgW * (textUnit->texImgBPP >> 3);
	textUnit->texImgData = new uchar[dataSize];
	memcpy(textUnit->texImgData, pImgData, dataSize);
}

void EyeballModel::setEyeballPos(const float* et)
{
	eyePos[0] = THU::Vector3f(et[0], et[1], et[2]);
	eyePos[1] = THU::Vector3f(et[3], et[4], et[5]);
	/*eyePos = std::vector<THU::Vector3f>(2);

#ifdef RENDER_SEPARATE_EYELID
	//eyePos[0] = THU::Vector3f(0.268510238, 0.471765947, 0.160450214);
	//eyePos[1] = THU::Vector3f(-0.268510238, 0.471765947, 0.160450214);
	eyePos[0] = THU::Vector3f(0.268510238, 0.481765947, 0.155450214);
	eyePos[1] = THU::Vector3f(-0.268510238, 0.481765947, 0.155450214);
#else
	eyePos[0] = THU::Vector3f(0.268510238, 0.471765947, 0.150450214);
	eyePos[1] = THU::Vector3f(-0.268510238, 0.471765947, 0.150450214);
#endif
	eyePos[0] += THU::Vector3f(et[0], et[1], et[2]);
	eyePos[1] += THU::Vector3f(et[3], et[4], et[5]);

	std::vector<THU::Vector3f> eyePosMesh(2);
	std::ifstream fin("data/eye_position_mesh.txt");
	fin >> eyePosMesh[0].x() >> eyePosMesh[0].y() >> eyePosMesh[0].z();
	fin >> eyePosMesh[1].x() >> eyePosMesh[1].y() >> eyePosMesh[1].z();
	fin.close();
	if (eyePosMesh[0].z() > 0 && eyePosMesh[1].z() > 0)
	{
		eyePos[0] = eyePosMesh[0];
		eyePos[1] = eyePosMesh[1];
	}
	else
	{
		std::ifstream fin("data/eye_position_mesh_fix.txt");
		fin >> eyePosMesh[0].x() >> eyePosMesh[0].y() >> eyePosMesh[0].z();
		fin >> eyePosMesh[1].x() >> eyePosMesh[1].y() >> eyePosMesh[1].z();
		fin.close();
		eyePos[0] = eyePosMesh[0] + THU::Vector3f(et[0], et[1], et[2]);
		eyePos[1] = eyePosMesh[1] + THU::Vector3f(et[3], et[4], et[5]);
	}
	PRINT_VALUE(eyePos[0].transpose());
	PRINT_VALUE(eyePos[1].transpose());*/
}

void EyeballModel::setPupilPos(const float* pp)
{
	memcpy(pupilPos[0].data(), pp, sizeof(float) * 3);
	memcpy(pupilPos[1].data(), pp + 3, sizeof(float) * 3);
}

void EyeballModel::updateTextureCordinates(float irisP)
{
	PRINT_INFO("Set iris size: " << irisP);
	float rate = irisPhi / irisP;
	std::vector<THU::Vector2f> vertexTexBuf;
	THU::Vector2f center(0.5, 0.5);
	eyeballMesh->getAllTexCoords_(vertexTexBuf);
	for (int i = 0; i < vertexTexBuf.size(); i++)
	{
		vertexTexBuf[i] = center + (vertexTexBuf[i] - center) * rate;
		if (vertexTexBuf[i].x() > 1)
			vertexTexBuf[i].x() = 1;
		if (vertexTexBuf[i].x() < 0)
			vertexTexBuf[i].x() = 0;
		if (vertexTexBuf[i].y() > 1)
			vertexTexBuf[i].y() = 1;
		if (vertexTexBuf[i].y() < 0)
			vertexTexBuf[i].y() = 0;
	}
	eyeballMesh->setVertexTexCoords_(vertexTexBuf);
	//eyeballMesh->setAllVerticesTexCoords(vertexTexBuf);
	//eyeballMesh->saveTexMesh("data/eyeball_id.obj");
	irisPhi = irisP;
}

void EyeballModel::updateEyeballSize(float radius)
{
	std::vector<THU::Vector3f> vtxBuf;
	eyeballMesh->getAllVertices(vtxBuf);
	int vertexNum = vtxBuf.size();
	float r = vtxBuf[0].norm();
	for (int i = 0; i < vertexNum; i++)
	{
		vtxBuf[i] = vtxBuf[i] / r * radius;
	}
	eyeballMesh->setVertices_(vtxBuf);
}

void EyeballModel::outputMesh(const float scale, const THU::Matrix3f& rot, const THU::Vector3f& trans, const std::string& label)
{
	for (int e = 0; e < 2; e++)
	{
		THU::TriMesh mesh = *eyeballMesh;
		std::vector<THU::Vector3f> vBuf;
		eyeballMesh->getAllVertices_(vBuf);

		for (int i = 0; i < vBuf.size(); i++)
		{
			vBuf[i] = scale * rot * (vBuf[i] + eyePos[e]) + trans;
		}
		mesh.setVertices(vBuf[0].data());
		mesh.saveMesh("eye" + std::to_string(e) + "_" + label + ".obj");
		//mesh.saveMesh("iteration/mesh/eye" + std::to_string(e) + "_" + label + ".obj");
	}
}

void EyeballModel::outputColorMesh(const float scale, std::vector<float> &phi, std::vector<float> &theta, const THU::Matrix3f& rot, const THU::Vector3f& trans, const std::string& label)
{
	for (int e = 0; e < 2; e++)
	{
		THU::Matrix3f eye_rot = phiTheta2Rotation(phi[e], theta[e]);
		THU::TriMesh mesh = *eyeballMesh;
		std::vector<THU::Vector3f> vBuf;
		std::vector<THU::Vector2f> tBuf;
		eyeballMesh->getAllVertices_(vBuf);
		eyeballMesh->getAllTexCoords_(tBuf);
		for (int i = 0; i < vBuf.size(); i++)
		{
			vBuf[i] = scale * rot * (eye_rot * vBuf[i] + eyePos[e]) + trans;
		}
		mesh.setVertices(vBuf[0].data());
		for (int i = 0; i < mesh.vertexNum(); i++)
		{
			int x = int((irisTex->texImgW - 1) * tBuf[i].x() + 0.5f);
			int y = int((irisTex->texImgH - 1) * tBuf[i].y() + 0.5f);
			mesh.setVertexColor(i, 
				irisTex->texImgData[4 * (irisTex->texImgW * y + x) + 2],
				irisTex->texImgData[4 * (irisTex->texImgW * y + x) + 1],
				irisTex->texImgData[4 * (irisTex->texImgW * y + x)]);
			//std::cout << x << " " << y << std::endl;
			//std::cout << mesh.vertexColor(i) << std::endl;
		}
		mesh.saveColorMesh(label + std::to_string(e) + ".off" );
	}
}

THU::Matrix3f EyeballModel::phiTheta2Rotation(float phi, float theta)
{
	THU::Matrix3f ret;
	THU::Matrix3f matY, matX;
	matY << 
		cos(theta), 0, sin(theta),
		0, 1, 0,
		-sin(theta), 0, cos(theta);
	matX << 
		1, 0, 0,
		0, cos(phi), -sin(phi),
		0, sin(phi), cos(phi);
	ret = matY * matX;
	return ret;
}