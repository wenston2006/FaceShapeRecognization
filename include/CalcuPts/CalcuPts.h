//
// Created by liusong on 17-9-20.
//

#ifndef FACERELATED_CALCUPTS_H
#define FACERELATED_CALCUPTS_H

#include <stdio.h>
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <functional>
#include "ncnn/net.h"
#include <time.h>
#include "abilix_core/face_image.h"
#include <iostream>
#include <fstream>

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

#endif //FACERELATED_CALCUPTS_H
