#pragma once
//#include <core/core.hpp>

namespace fmark  //face land mark
{
	bool init(std::string&strlandmarkpath);

	void shape_landmark (const cv::Mat& img,int lt_x,int lt_y,int width,int heigh,std::vector<cv::Point2f>& facekeypts);

//extern std::string strlandmarkpath;
};
