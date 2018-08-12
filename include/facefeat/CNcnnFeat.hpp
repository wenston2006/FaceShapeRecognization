#ifndef _ABILIX_CNCNN_FEAT_H_
#define _ABILIX_CNCNN_FEAT_H_

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <functional>
#include "ncnn/net.h"
#include <time.h>
#include "infilixcore/face_image.h"
#include "CFaceFeat.hpp"


class CNcnnFeat :public CCalFaceFeat
{
public:
	CNcnnFeat();
	virtual ~CNcnnFeat();
	virtual bool extractDescriptors(FaceImgData& rawgrayimg, int nIdRect);
	
	std::vector<cv::Point2f>& getvalidfacekeypts();
	
	virtual cv::Mat& get_img(){
		return standard_img;
	}

	virtual int init(std::string sPathLandmarkModel);

	virtual int get_feats(float *pfeats);

	virtual int destroy();
private:
	//std::vector<cv::Point2f> validfacekeypts;
	//std::vector<cv::Point2f> facekeypts;	//sum :68	

	cv::Mat standard_img;

	int gen_standard_img(FaceImgData& rawgrayimg, int nIdRect);
	int calacDescriptors();

	int trans2standardImg(FaceImgData& rawgrayimg);
	void updatavalidkeypts();
	void extractvalidkeypts();

	float *m_feats;
};

class CCalFeat
{
public:
	CCalFeat();
	~CCalFeat(){
		if (NULL != m_feats)
		{
			delete[] m_feats;
			m_feats = NULL;
		}
	};

	int init(std::string sPathModel);
	int calacDescriptors(unsigned char * psAlignface);
	int get_feats(float *pfeats);

private:
	ncnn::Net squeezenet;
	float *m_feats;
};

class CCalPts
{
public:
	CCalPts();
	~CCalPts(){
	};

	int init(std::string sPathLandmarkModel);

	int GetLandmark(cv::Mat cSrcImg, cv::Rect cRect, std::vector<cv::Point2f> &facekeypts);

private:
	ncnn::Net squeezenet_landmark;
};

int FaceProcessInit(const char *sModelPath);
int FaceProcessDestroy();

#endif
