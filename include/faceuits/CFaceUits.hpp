#pragma once
//#include "core/person.h"
#include "infilixcore/person.h"
#include <string>
#include <vector>

class CFaceUit
{
public:
	//static void setBands(int wband,int hband);
	//static int init(float th, int scale, int w, int h, int nchns, std::string& dbdir, std::string& facedir);
	static int init(float th, int scale, std::string& modeldir, std::string& facedir);
	static int setImageSize(int w, int h, int nchns);
	//static bool facedetect(const cv::Mat& rgbimg,int xy[4]);
	static bool facedetect(const unsigned char* imgdata,int xy[100]);

	static bool addface(int personid);
/*
	static void clearfaces();
	static bool addface();
	static bool delface();
	static int addperson(int id);
	*/
	static int delperson(int id);
	static bool person_exist(int id);
	static int facepredict(int anLabels[]);
	static int destory();
	//static int getpersonList(std::vector<std::string> &personList);
	static int getpersonList(std::vector<int> &personList);
	static int facepredict4mul(float anLabels[]);
private:
	CFaceUit(const CFaceUit & rhs);
	CFaceUit & operator = (const CFaceUit & rhs);
};
