#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include "opencv2/opencv.hpp"
#include "ArgumentParser.h"
#include "faceuits/CFaceUits.hpp"

#ifdef WIN32
#include <io.h> //use _findfirst() and findnext() function
#include <shlwapi.h>
#include <string>

#pragma comment(lib, "shlwapi.lib")
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif // WIN32

using namespace std;
using namespace cv;

const map<string, string> defaultArguments = { { "-o", "\0" } };

#define IMG_TEMP_WIDTH				160
#define IMG_TEMP_HEIGHT				160
#define TOTALDIM_NUM				75520
#define MAX_PATH_LEN				1024
#define FEAT_LENGTH_AFTER_REDUCT	527
#define MAX_NUM_FACES_MAIN			100
#define MAX_IMAGE_WIDTH				1920
#define MAX_IMAGE_HEIGHT			1216
//#define OUTPUT_IMAGE_NAME_LIST
//#define OUTPUT_FEATURE
//#define OUTPUT_IMAGE_LABEL_LIST

unsigned char aImageData[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT * 3];

const string usage = string("Usage: *.exe [options] \n"
	"options:\n"
	"-p 1 -i [personid] -m [path2model] -o [path2imagelib] [path2inputImage]: add face to image data lib \n"
	"-p 2 -i [personid] -m [path2model] -o [path2imagelib]: remove face images of person with id \n"
	"-p 3 [path2inputImage] -m [path2model] -o [path2imagelib]: predict the id of the input image \n"
	"to set threshold during prediction, use:\n"
	"-p 3 [path2inputImage] -m [path2model] -t [threshold value] -o [path2imagelib]: predict the id of the input image \n"
	"-p 4  -m [path2model] -o [path2imagelib]: get all ids in the image data lib \n");

static void set_image(unsigned char *pImageData, cv::Mat SrcImg)
{
	int nchns = SrcImg.channels();
	if (1 == nchns)
	{
		memcpy(pImageData, SrcImg.data, sizeof(unsigned char)* SrcImg.rows * SrcImg.cols);
	}
	else if (3 == nchns)
	{
		for (int h = 0; h < SrcImg.rows; h++)
		{
			for (int w = 0; w < SrcImg.cols; w++)
			{
				pImageData[h*SrcImg.cols*nchns + w*nchns + 0] = SrcImg.at<Vec3b>(h, w)[0];
				pImageData[h*SrcImg.cols*nchns + w*nchns + 1] = SrcImg.at<Vec3b>(h, w)[1];
				pImageData[h*SrcImg.cols*nchns + w*nchns + 2] = SrcImg.at<Vec3b>(h, w)[2];
			}
		}
	}
}

int main(int argc, const char **argv)
{
	//double d64T1, d64T2;
	//double d64Tmp = 0;
	//double d64PeakTime = -9999;
	//double d64Time1 = 0;

	int l32SignAddFace = 0;
	int l32SignDelFace = 0;
	int l32SignPredict = 0;
	int l32SignGetList = 0;

	int xy[5*MAX_NUM_FACES_MAIN+1];
	int nStatus;
	int aStatus;
	bool bStatus;
	bool bFaceDet;
	string sMsg;
	std::vector<int> vPersonList;
	string sId;
	int nPersonid;
	int *feature = new int[TOTALDIM_NUM];
	float *pfFeatDst = new float[FEAT_LENGTH_AFTER_REDUCT];
	float fThreshold = 0.80f; // 0.70f;// 0.65f;
    int nLabel = -1;
	cv::Mat image, gray, imageB, imageE;
	std::string s8ModelPath;

   ArgumentParser argParser(argc, argv, usage, defaultArguments);
   vector<string> inputList = argParser.getInputList();

   std::string sThreshold = argParser.getArgument("-t");
   if (sThreshold.size() > 0)
   {
	   fThreshold = stof(sThreshold);
   }

    std::string sLabel = argParser.getArgument("-L");
    if (sLabel.size() > 0)
    {
        nLabel = stoi(sLabel);
    }

   std::string sImageLibPath = argParser.getArgument("-o");
   if (0 == sImageLibPath.size())
   {
	   argParser.printUsage();
	   nStatus = -1;
	   goto Exit;
	   //return -1;
   }

	s8ModelPath = argParser.getArgument("-m");
	if (0 == s8ModelPath.size())
	{
		argParser.printUsage();
		nStatus = -1;
		goto Exit;
		//return -1;
	}

   //get all image list and generate the corresponding labels
   if (argParser.getArgument("-p") == "1")
   {
	   l32SignAddFace = 1;
   }
   else if (argParser.getArgument("-p") == "2")
   {
	   l32SignDelFace = 1;
   }
   else if (argParser.getArgument("-p") == "3")
   {
	   l32SignPredict = 1;
   }
   else if (argParser.getArgument("-p") == "4")
   {
	   l32SignGetList = 1;
   }
   if (1 == l32SignAddFace || 1 == l32SignDelFace)
   {
	   sId = argParser.getArgument("-i");
	   nPersonid = atoi(sId.c_str());
   }

   if (1)
   {
	   if (1 == l32SignAddFace || 1 == l32SignPredict)
	   {
		   if (inputList.size() == 0) {
			   argParser.printUsage();
			   nStatus = -1;
			   goto Exit;
		   }

		   image = cv::imread(inputList[0].c_str(), 1);
#if 0
		   cv::imshow("resized", image);
		   cv::waitKey(0);
#endif

#if 0
		   image = cv::imread(inputList[0].c_str(), 0);
#endif
		   if (NULL == image.data)
		   {
			   nStatus = -1;
			   goto Exit;
		   }
#if 1
		   int unifcols = (image.cols / 4) * 4;
		   int unifrows = (image.rows / 4) * 4;

//           int unifcols = (image.cols / 10);
//           int unifrows = (image.rows / 10);
		   cv::resize(image, image, cv::Size(unifcols, unifrows));

#if 0
           cv::imshow("BGR3", image);
           cv::waitKey(0);
#endif

#if 0
		   cv::cvtColor(image, gray, CV_BGR2GRAY);
		   gray = gray.clone();
#endif

#if 0
		   cv::imshow("resized", gray);
		   cv::waitKey(0);
#endif

#endif

#if 0
		   printf("Input name is: %s\n", inputList[0].c_str());
#endif

		   CFaceUit::setImageSize(image.cols, image.rows, 3);
		   CFaceUit::init(fThreshold, 4, s8ModelPath, sImageLibPath);
		   bFaceDet = CFaceUit::facedetect(image.data, xy);
		   if (true == bFaceDet)
		   {
			   nStatus = 0;
		   }
		   else if (false == bFaceDet)
		   {
			   nStatus = -1;
			   goto Exit;
		   }

	   }
	   else if (1 == l32SignGetList || 1 == l32SignDelFace)
	   {
		   if (0 != nStatus)
		   {
			   goto Exit;
		   }
	   }

	   if (1 == l32SignPredict)
	   {
		   float anLabels[MAX_NUM_FACES_MAIN];
		   memset(anLabels, -1, sizeof(float)*MAX_NUM_FACES_MAIN);
		   aStatus = CFaceUit::facepredict4mul(anLabels);
		   if (true == bFaceDet)
		   {
			   //sMsg = "success";
			   nStatus = 0;
			   //for (int ii = 0; ii < 1; ii++)
			   //{
				  // printf("faceshape pre: %d\n", as8ImageFile, (int)(anLabels[ii*FACE_PROPERTY_SIZE + 1]));
			   //}

		   }
		   else if (false == bFaceDet)
		   {
			   //sMsg = "others";
			   nStatus = -1;
			   goto Exit;
		   }		  
	   }
	   
	   CFaceUit::destory();
   }
   return 0;
Exit:
   char sTmp[1024];
   sprintf(sTmp, "%d", nStatus + 110);
   string sCode = string("{\"code\":") + string(sTmp) + ",";
   string sMsgs = string("\"msg\":\"others\"}");
   cout << sCode << sMsgs << endl;
   return 0;
}
