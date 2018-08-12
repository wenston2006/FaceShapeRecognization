#ifndef _ABILIX_FACIAL_LANDMARK_TOOL_H_
#define _ABILIX_FACIAL_LANDMARK_TOOL_H_

#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;

template<typename T>
T MIN_ABILIX(T a, T b)
{
	return   ((a) > (b) ? (b) : (a));
}

template<typename T>
T MAX_ABILIX(T a, T b)
{
	return   ((a) > (b) ? (a) : (b));
}

template<typename T>
T ABS_ABILIX(T x)
{
	return ((x) >= 0 ? (x) : (-(x)));
}

int FacialLandmarkCheck(std::vector<cv::Point2f> facekeypts, int NumKeypts, int nImgWidth, int nImgHeight)
{
	int nStatus = 0;
	for (int ii = 0; ii < NumKeypts; ii++)
	{
		cv::Point2f SinglePts = facekeypts[ii];
#if 1
		if ( SinglePts.x < 0 || (SinglePts.x - nImgWidth) > 1e-5)
		{
			//nStatus = -1;
			//return nStatus;
			SinglePts.x = MAX_ABILIX(0, MIN_ABILIX(int(SinglePts.x), nImgWidth));
		}
		if (SinglePts.y < 0 || (SinglePts.y - nImgHeight) > 1e-5)
		{
			//nStatus = -1;
			//return nStatus;
			SinglePts.y = MAX_ABILIX(0, MIN_ABILIX(int(SinglePts.y), nImgHeight));
		}
#endif	
		
	}
	return 0;
}



#endif