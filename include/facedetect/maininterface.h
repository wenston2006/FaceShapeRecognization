#ifndef _MAIN_INTERFACE_H_
#define _MAIN_INTERFACE_H_

//#include <core\core.hpp>

namespace fadt{

	bool haarinit();

//	void haarfacedetect(const cv::Mat&image,int scale=4);
	void haarfacedetect(const unsigned char*pimage,int _width,int _height,int scale=4);
};


#endif

