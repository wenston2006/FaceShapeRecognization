#ifndef _INFILIX_CFACE_FEAT_H_
#define _INFILIX_CFACE_FEAT_H_

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "infilixcore/face_image.h"

#ifdef FACEFEAT_EXPORTS
#define FACEFEAT_API __declspec(dllexport)
#else
#define FACEFEAT_API __declspec(dllimport)
#endif

//#ifdef __cplusplus    //__cplusplus是双下划线
//extern "C" {
//#endif

class CCalFaceFeat
{
public:
	FACEFEAT_API static CCalFaceFeat* create();	
	CCalFaceFeat();
	virtual ~CCalFaceFeat();
	virtual bool extractDescriptors(FaceImgData& rawgrayimg, int nIdRect) = 0;
	virtual cv::Mat& get_img() = 0;
	virtual int init(std::string sPathLandmarkModel) = 0;
	virtual int get_feats(float *pfeats) = 0;
	virtual int destroy() = 0;
};

FACEFEAT_API int destroyG(CCalFaceFeat* cFeatSp);

//#ifdef __cplusplus
//}
//#endif

#endif