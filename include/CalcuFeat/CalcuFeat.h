//
// Created by liusong on 17-9-20.
//

#ifndef FACERELATED_CALCUFEAT_H
#define FACERELATED_CALCUFEAT_H

#include <stdio.h>
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <functional>
#include "ncnn/net.h"
#include <time.h>
#include <iostream>
#include <fstream>


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

#endif //FACERELATED_CALCUFEAT_H
