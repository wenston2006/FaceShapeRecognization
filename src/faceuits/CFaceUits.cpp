#include "CFaceUits.hpp"
#include "infilixutils/diskmanager.h"
#include "facedetect/CFaceTrack.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include "infilixcore/CFaceComp.h"
#include "facefeat/CFaceFeat.hpp"

#include "infilixLandmark/faciallandmark.h"

#define MAX_NUM_FACES								100
#define MAX_NUM_FEATS								7 //17*2+1 //512
#define N_FACE_SHAPE                                4
#define MAX_IMAGE_WIDTH                             1000
#define MAX_IMAGE_HEIGHT                            1000
#define MAX_LIST_SIZE                               5
#define OUTPUT_ARRAY_STEP                           11
#define MAX_NOFACE_LASTING                          1

const float g_thresholds[4] = { 0.70f, 0.70f, 0.75f, 0.9f };

using namespace cv;
using namespace std;

typedef struct
{
	int nId;
	std::vector<std::pair<int, int>> vTestPerson;
}tPersonShoot;

typedef struct
{
	int channels;
	int rows;
	int cols;
	unsigned char *data;
}tBImage;

static std::vector<person*> personslist;

static std::list<float> gEllipseList;
static std::list<float> gRoundList;
static std::list<float> gSquareList;
static std::list<float> gTriangleList;

static float g_thresh=0.38f;
static int g_scale=4;
static int g_nchns;
static int g_width;
static int g_height;
static int g_GetOneFace = 1;

tBImage curr_gray_img;//the image is rote transed
tBImage curr_color_img;

cv::Point lef_top[MAX_NUM_FACES], rig_down[MAX_NUM_FACES];
std::string sImageDBdir;
static int g_NumFaceDet;
static int g_nCntFaceDet;
static FaceImgData faimg;
static CCalFaceFeat *pcCalFeatExtracrt;

#define debug

#ifdef debug
#define logF(agr) logfile<<agr<<"\n"; \
		logfile.flush();
#else
#define logF(agr)
#endif

std::ofstream logfile("/mnt/sdcard/facesDb/face_logs.txt",std::ios::out|std::ios::ate);

bool getface(FaceImgData& faimg, int nSMulFaceGet);

inline void int2string(int id,std::string& outstr)
{
	char temp[64]={0};
	sprintf(temp, "%d", id);
	std::string str_id(temp);
	outstr=str_id;
}

inline int string2int(const char*str)
{
	return atoi(str);
}
void load_persons_data(std::string imageDBdir)
{
	std::vector<std::string> personnames;
	std::string dir = imageDBdir;
	UTIL::CDir::GetFoldersInDirs(dir,personnames);
	std::vector<std::string>::iterator itr=personnames.begin();

	while (itr!=personnames.end()){
		person*pren=new person(*itr++);
		pren->load_faces_data(imageDBdir);
		personslist.push_back(pren);
	}
}
void getcurrentpath(std::string& fullpath)
{
	fullpath="/mnt/sdcard/facesDb";
}

int CFaceUit::init(float th, int scale, std::string& modeldir, std::string& facedir)
{
	bool bStatus;
	int retg;
	g_thresh = th;

	//get current path;
	std::string currstrPath = modeldir;
	sImageDBdir = facedir;

	logF(currstrPath);

#if 1
	UTIL::CDir::CreatDir(facedir.c_str());
#endif

#if 1
	load_persons_data(facedir);
#endif
	// init haar
    std::string sAddrModel = currstrPath;
#if 1
    int ret = facepose2init(g_width, g_height, sAddrModel);
	if (ret != 0)
	{
		//printf("facetrack init failed!\n");
		return -1;
	}
#endif

	//initialize CCalfeat
	pcCalFeatExtracrt = CCalFaceFeat::create();
	if (NULL == pcCalFeatExtracrt)
	{
		return -2;
	}
	ret = pcCalFeatExtracrt->init(currstrPath);
	if (0 != ret)
	{
		return -3;
	}

	return 0;
}

int CFaceUit::setImageSize(int w, int h, int nchns)
{
	g_nchns = nchns;
    g_width = w;
    g_height = h;
	if (1 == nchns)
	{
#if 1
		curr_gray_img.rows = h;
		curr_gray_img.cols = w;
		curr_gray_img.channels = 1;
		curr_gray_img.data = new unsigned char[h * w];
#endif
	}
	else if (3 == nchns)
	{
		curr_gray_img.rows = h;
		curr_gray_img.cols = w;
		curr_gray_img.channels = 1;
		curr_gray_img.data = new unsigned char[h * w];

		curr_color_img.rows = h;
		curr_color_img.cols = w;
		curr_color_img.channels = 3;
		curr_color_img.data = new unsigned char[h * w * 3];
	}
	faimg.init(h, w);
	return 0;
}

static void copyimg(const unsigned char* imgdata,cv::Mat& gray_img)
{
	int nchns = gray_img.channels();
	if (1 == nchns)
	{
		for (int h = 0; h < gray_img.rows; h++)
		for (int w = 0; w < gray_img.cols; w++)
		{
			gray_img.at<unsigned char>(h, w) = imgdata[h*gray_img.cols + w];
		}
	}
	else if (3==nchns)
	{
		for (int h = 0; h < gray_img.rows; h++)
		{
			for (int w = 0; w < gray_img.cols; w++)
			{
				gray_img.at<Vec3b>(h, w)[0] = imgdata[h*gray_img.cols*nchns + w*nchns + 0];
				gray_img.at<Vec3b>(h, w)[1] = imgdata[h*gray_img.cols*nchns + w*nchns + 1];
				gray_img.at<Vec3b>(h, w)[2] = imgdata[h*gray_img.cols*nchns + w*nchns + 2];
			}
		}
	}
}

static void copyimg(const cv::Mat& gray_img, unsigned char* imgdata)
{
	int nchns = gray_img.channels();
	if (1 == nchns)
	{
		for (int h = 0; h < gray_img.rows; h++)
		for (int w = 0; w < gray_img.cols; w++)
		{
			imgdata[h*gray_img.cols + w] = gray_img.at<unsigned char>(h, w);
		}
	}
	else if (3 == nchns)
	{
		for (int h = 0; h < gray_img.rows; h++)
		{
			for (int w = 0; w < gray_img.cols; w++)
			{
				imgdata[h*gray_img.cols*nchns + w*nchns + 0] = gray_img.at<Vec3b>(h, w)[0];
				imgdata[h*gray_img.cols*nchns + w*nchns + 1] = gray_img.at<Vec3b>(h, w)[1];
				imgdata[h*gray_img.cols*nchns + w*nchns + 2] = gray_img.at<Vec3b>(h, w)[2];
			}
		}
	}
}

static void copyimg(const unsigned char* imgdata, tBImage & gray_img)
{
	int nchns = gray_img.channels;
	if (1 == nchns)
	{
		for (int h = 0; h < gray_img.rows; h++)
		{
			for (int w = 0; w < gray_img.cols; w++)
			{
				gray_img.data[h*gray_img.cols + w] = imgdata[h*gray_img.cols + w];
			}
		}
	}
	else if (3 == nchns)
	{
		for (int h = 0; h < gray_img.rows; h++)
		{
			for (int w = 0; w < gray_img.cols; w++)
			{
				gray_img.data[h*gray_img.cols*nchns + w*nchns + 0] = imgdata[h*gray_img.cols*nchns + w*nchns + 0];
				gray_img.data[h*gray_img.cols*nchns + w*nchns + 1] = imgdata[h*gray_img.cols*nchns + w*nchns + 1];
				gray_img.data[h*gray_img.cols*nchns + w*nchns + 2] = imgdata[h*gray_img.cols*nchns + w*nchns + 2];
			}
		}
	}
}

static void copyimg(const tBImage & gray_img, unsigned char* imgdata)
{
	int nchns = gray_img.channels;
	if (1 == nchns)
	{
		for (int h = 0; h < gray_img.rows; h++)
		{
			for (int w = 0; w < gray_img.cols; w++)
			{
				imgdata[h*gray_img.cols + w] = gray_img.data[h*gray_img.cols + w];
			}
		}
	}
	else if (3 == nchns)
	{
		for (int h = 0; h < gray_img.rows; h++)
		{
			for (int w = 0; w < gray_img.cols; w++)
			{
				imgdata[h*gray_img.cols*nchns + w*nchns + 0] = gray_img.data[h*gray_img.cols*nchns + w*nchns + 0];
				imgdata[h*gray_img.cols*nchns + w*nchns + 1] = gray_img.data[h*gray_img.cols*nchns + w*nchns + 1];
				imgdata[h*gray_img.cols*nchns + w*nchns + 2] = gray_img.data[h*gray_img.cols*nchns + w*nchns + 2];
			}
		}
	}
}


bool CFaceUit::facedetect(const unsigned char* imgdata,int xy[100])
{
	int ret;
	int nOutNumFace = 0;
#define MAXNDETECTIONS 2048					
	float frect[5 * MAXNDETECTIONS] = { 0 };
	cv::Mat cGrayImg;
	cv::Mat cColorImg;

//#define	TEST_ANDROID_VERSION
#ifdef TEST_ANDROID_VERSION 
#define	ROTATE_180
#ifdef ROTATE_90
	//cColorImg = cv::Mat(curr_color_img.cols*1.5, curr_color_img.rows, CV_8UC1, (unsigned char*)imgdata);
#elif defined ROTATE_180
	cColorImg = cv::Mat(curr_color_img.rows*1.5, curr_color_img.cols, CV_8UC1, (unsigned char*)imgdata);
#endif

#if 1
	imshow("gary1", cColorImg);
	waitKey(0);
#endif
	cv::Mat tempbgr, tempbgrT, tempbgrTc;
	//cv::cvtColor(cColorImg, tempbgr, CV_YUV2BGR_NV21); 
	cv::cvtColor(cColorImg, tempbgr, CV_YUV2BGR_I420);
#if 1
	imshow("gary2", tempbgr);
	waitKey(0);
#endif

#ifdef  ROTATE_180
	cv::flip(tempbgr, tempbgr, 0);
	tempbgrTc = tempbgr.clone();
#elif defined ROTATE_90
	cv::flip(tempbgr, tempbgr, 1);
	cv::transpose(tempbgr, tempbgrT);
	tempbgrTc = tempbgrT.clone();
#endif

	//cColorImg = cv::Mat(curr_color_img.rows, curr_color_img.cols, CV_8UC3, curr_color_img.data);
	copyimg(tempbgrTc.data, curr_color_img);
#if 1
	imshow("gary", tempbgrTc);
	waitKey(0);
#endif	
	cv::cvtColor(tempbgrTc, cGrayImg, CV_BGR2GRAY);

#else
	copyimg(imgdata, curr_color_img);
	cColorImg = cv::Mat(curr_color_img.rows, curr_color_img.cols, CV_8UC3, curr_color_img.data);
	cv::cvtColor(cColorImg, cGrayImg, CV_BGR2GRAY);
#endif
	memcpy(curr_gray_img.data, cGrayImg.data, sizeof(unsigned char)* curr_color_img.rows * curr_color_img.cols);
#if 0
	Mat ctmp = cv::Mat(curr_gray_img.rows, curr_gray_img.cols, CV_8UC1, curr_gray_img.data);
	imshow("gary", ctmp);
	waitKey(0);
#endif

#if 1
	if (curr_gray_img.data)
	{
        ret = facepose2detect((char*)curr_gray_img.data, frect);
		if (0 != ret)
		{
			g_nCntFaceDet++;
#if 1
			printf("******************************\n");
			printf("face detection failed branch g_nCntFaceDet is %d\n", g_nCntFaceDet);
			printf("******************************\n");
#endif

#if 1
			if (g_nCntFaceDet >= MAX_NOFACE_LASTING)
			{
				gEllipseList.clear();
				gRoundList.clear();
				gSquareList.clear();
				gTriangleList.clear();
#if 1
				printf("***********************\n");
				printf("max no face lasting occurred!\n");
				printf("***********************\n");
#endif
				g_nCntFaceDet = 0;
			}
#endif

			return false;
		}
	}
	else
	{
		logF("faimg.faceGrayImg.data empty!");
		return false;
	}
#endif

	cv::Rect tRect;
#if 1
    nOutNumFace = frect[0];
	if (nOutNumFace > 0)
	{
		xy[0] = 0;
		//\B6\D4\C8\CB\C1\B3\B3ߴ\E7\BD\F8\D0\D0\C5\C5\D0\F2\A3\AC\D7\EE\B4\F3\B5\C4\C8\CB\C1\B3\B7\C5\D4\DA\D7\EE\C9\CF\C3\E6
		int l32MaxHeight = -999;
		int l32IndexMax = -1;
		int l32Index = 0;

		if (0 == g_GetOneFace) {
			for (int l32Index0 = 0; l32Index0 < nOutNumFace; l32Index0++) {
				lef_top[l32Index].x = frect[5 * l32Index0 + 0 + 1];
				lef_top[l32Index].y = frect[5 * l32Index0 + 1 + 1];
				rig_down[l32Index].x = frect[5 * l32Index0 + 2 + 1] + frect[5 * l32Index0 + 0 + 1];
				rig_down[l32Index].y = frect[5 * l32Index0 + 3 + 1] + frect[5 * l32Index0 + 1 + 1];
				xy[l32Index * 5 + 0 + 1] = frect[5 * l32Index0 + 0 + 1];
				xy[l32Index * 5 + 1 + 1] = frect[5 * l32Index0 + 1 + 1];
				xy[l32Index * 5 + 2 + 1] = frect[5 * l32Index0 + 2 + 1] + xy[l32Index * 5 + 0 + 1];
				xy[l32Index * 5 + 3 + 1] = frect[5 * l32Index0 + 3 + 1] + xy[l32Index * 5 + 1 + 1];
				xy[l32Index * 5 + 4 + 1] = frect[5 * l32Index0 + 4 + 1];
				l32Index++;
				xy[0]++;
			}
		}
		else if (1 == g_GetOneFace)
		{
			for (int l32Index0 = 0; l32Index0 < nOutNumFace; l32Index0++)
			{
				if (frect[5 * l32Index0 + 3 + 1] > l32MaxHeight)
				{
					l32MaxHeight = frect[5 * l32Index0 + 3 + 1];
					l32IndexMax = l32Index0;
				}
			}
			lef_top[l32Index].x = frect[5 * l32IndexMax + 0 + 1];
			lef_top[l32Index].y = frect[5 * l32IndexMax + 1 + 1];
			rig_down[l32Index].x = frect[5 * l32IndexMax + 2 + 1] + frect[5 * l32IndexMax + 0 + 1];
			rig_down[l32Index].y = frect[5 * l32IndexMax + 3 + 1] + frect[5 * l32IndexMax + 1 + 1];
			xy[l32Index * 5 + 0 + 1] = frect[5 * l32IndexMax + 0 + 1];
			xy[l32Index * 5 + 1 + 1] = frect[5 * l32IndexMax + 1 + 1];
			xy[l32Index * 5 + 2 + 1] = frect[5 * l32IndexMax + 2 + 1] + xy[l32Index * 5 + 0 + 1];
			xy[l32Index * 5 + 3 + 1] = frect[5 * l32IndexMax + 3 + 1] + xy[l32Index * 5 + 1 + 1];
			xy[l32Index * 5 + 4 + 1] = frect[5 * l32IndexMax + 4 + 1];
			l32Index++;
			xy[0] = 1;
		}
#if 1
		cv::Rect tRect;
		tRect.x = lef_top[l32Index - 1].x;
		tRect.y = lef_top[l32Index - 1].y;
		tRect.width = rig_down[l32Index - 1].x - lef_top[l32Index - 1].x;
		tRect.height = rig_down[l32Index - 1].y - lef_top[l32Index - 1].y;

		cv::rectangle(cColorImg, tRect, cv::Scalar(0, 255, 0));
		cv::imshow("imgp_facedetected", cColorImg);
		//cv::rectangle(tempbgrTc, tRect, cv::Scalar(0, 255, 0));
		//cv::imshow("imgp_facedetected", tempbgrTc);
		cv::waitKey(0);
#endif
	}
#endif

		g_NumFaceDet = xy[0];
		if (g_NumFaceDet <= 0)
		{
			g_nCntFaceDet++;
#if 1
			printf("******************************\n");
			printf("false branch g_nCntFaceDet is %d\n", g_nCntFaceDet);
			printf("******************************\n");
#endif

#if 1
			if (g_nCntFaceDet >= MAX_NOFACE_LASTING)
			{
				gEllipseList.clear();
				gRoundList.clear();
				gSquareList.clear();
				gTriangleList.clear();
#if 1
				printf("***********************\n");
				printf("max no face lasting occurred!\n");
				printf("***********************\n");
#endif
				g_nCntFaceDet = 0;
			}
#endif
			return false;
		}
		else
		{
#if 0
			printf("******************************\n");
			printf("true branch g_nCntFaceDet is %d\n", g_nCntFaceDet);
			printf("******************************\n");
#endif
			return true;
		}	
}

bool CFaceUit::addface(int personid)
{
#if 1
	if(person_exist(personid)){
		delperson(personid);
	}
#endif
	float aFeats[MAX_NUM_FEATS];
	faimg.clear();
	getface(faimg,0);

	CFace*pface=new CFace();	
	if(!pcCalFeatExtracrt->extractDescriptors(faimg,0))
	{
		printf("Failed feature extraction!");
		delete pface;
		return false;
	}
	else
	{
		printf("Finished feature extraction!");
		pcCalFeatExtracrt->get_feats(aFeats);
		pface->copy_feat(aFeats, MAX_NUM_FEATS);
		cv::Mat cStandardImg = pcCalFeatExtracrt->get_img();
		pface->set_img(cStandardImg);
	}
	std::string strname;
	int2string(personid,strname);
	for (int mg=0;mg<personslist.size();mg++)
	{
		if (!strname.compare(personslist[mg]->get_name()))
		{
			personslist[mg]->addface(pface, sImageDBdir, 1);
			return true;
		}
	}
	person* pron=new person(strname);
	pron->addface(pface, sImageDBdir, 1);
	personslist.push_back(pron);

	return true;
}

int CFaceUit::destory()
{
	int nStatus;
	for (int n = 0; n < personslist.size(); n++)
	{
		if (NULL != personslist[n])
		{
			personslist[n] = NULL;
			delete personslist[n];
		}		
	}			
	personslist.clear();
	faimg.destroy();

    nStatus = facepose2destroy();
	if (0 != nStatus)
	{
		//printf("face detector destroy failed!\n");
		return -1;
	}
	
	nStatus = pcCalFeatExtracrt->destroy();
	if (0 != nStatus)
	{
		//printf("Face process destroy failed!\n");
		return -1;
	}

	if (NULL != pcCalFeatExtracrt)
	{
		delete pcCalFeatExtracrt;
		pcCalFeatExtracrt = NULL;
	}
	return 0;
}

bool getface(FaceImgData& faimg, int nSMulFaceGet)
{
	if(!curr_gray_img.data) {
		logF("capture retrieve image error!");
		return false;
	}

#if 0
	cv::Mat cGrayImg;
	cv::Mat cColorImg;
	cColorImg = cv::Mat(curr_color_img.rows, curr_color_img.cols, CV_8UC3, curr_color_img.data);
	cv::cvtColor(cColorImg, cGrayImg, CV_BGR2GRAY);
	memcpy(curr_gray_img.data, cGrayImg.data, sizeof(unsigned char)* curr_color_img.rows * curr_color_img.cols);
#endif

	if (3 == g_nchns)
	{
		copyimg(curr_color_img, faimg.pColor);
		copyimg(curr_gray_img, faimg.pGray);
		faimg.nSignColor = 1;
	}
	else if (1 == g_nchns)
	{
		copyimg(curr_gray_img, faimg.pGray);
		faimg.nSignColor = 0;
	}
	if (0 == nSMulFaceGet)
	{
		cv::Rect tRect;
		//\B6\D4\C8\CB\C1\B3\B3ߴ\E7\BD\F8\D0\D0\C5\C5\D0\F2\A3\AC\D7\EE\B4\F3\B5\C4\C8\CB\C1\B3\B7\C5\D4\DA\D7\EE\C9\CF\C3\E6
		int l32MaxHeight = -999;
		int l32IndexMax = -1;
		for (int l32Index0 = 0; l32Index0 < g_NumFaceDet; l32Index0++)
		{
			if ((rig_down[l32Index0].y - lef_top[l32Index0].y) > l32MaxHeight)
			{
				l32MaxHeight = (rig_down[l32Index0].y - lef_top[l32Index0].y);
				l32IndexMax = l32Index0;
			}
		}
		int l32Index = l32IndexMax;
		tRect.x = lef_top[l32Index].x;
		tRect.y = lef_top[l32Index].y;
		tRect.width = rig_down[l32Index].x - lef_top[l32Index].x;
		tRect.height = rig_down[l32Index].y - lef_top[l32Index].y;
		faimg.rc_faceframe.clear();
		faimg.rc_faceframe.push_back(tRect);
	}
	else if (1 == nSMulFaceGet)
	{
		faimg.rc_faceframe.clear();
		cv::Rect tRect;
		for (int l32Index0 = 0; l32Index0 < g_NumFaceDet; l32Index0++)
		{
			tRect = cv::Rect(cv::Point(lef_top[l32Index0].x, lef_top[l32Index0].y), cv::Point(rig_down[l32Index0].x, rig_down[l32Index0].y));
			faimg.rc_faceframe.push_back(tRect);
		}			
	}	
	return true;	
}

int CFaceUit::delperson(int id)
{
	std::string strname;
	int2string(id,strname);
	std::vector<person*>::iterator itr=personslist.begin();
	for(;itr!=personslist.end();itr++)
	{
		if (!strname.compare((*itr)->get_name()))
		{
			std::string perfulldir = sImageDBdir + (*itr)->get_name();
			delete (*itr);
			personslist.erase(itr);
			// delete data from dir
			UTIL::CDir::deleteFolder(perfulldir);
			return id;
		}
	}
	return -1;
}
bool CFaceUit::person_exist(int personid)
{
	std::string strname;
	int2string(personid,strname);
	for (int mg=0;mg<personslist.size();mg++)
	{
		if (!strname.compare(personslist[mg]->get_name()))
			return true;
	}
	return false;
}
/*
//return false ,no find the face in data
bool predict(CFace* pface,std::string&result_name,double& dis)
{
	std::vector< std::pair<std::string,double> >resurest;

	search_face_object(*pface,personslist,resurest);

	if(resurest.size()>0)
	{
		// find the min dis
		int index=0;
		double maxconf=resurest[0].second;//[index];
		for (int i=0;i<resurest.size();i++)
		{
			if(resurest[i].second>maxconf){
				index=i;
				maxconf=resurest[i].second;
			}
		}

#if 1
		for (int i = 0; i<resurest.size(); i++)
		{
			printf("name %s, conf: %f\n", resurest[i].first.c_str(), float(resurest[i].second));
		}
#endif
		if (maxconf>g_thresh)
		{
			result_name=resurest[index].first;
			dis=resurest[index].second;
			return true;
		}
		else
			return false;

	}
	return false;
}
*/

/*
//return false ,no find the face in data
bool predict4mul(CFace* pface, tPersonCmp *pvPersonDist)
{
	std::vector< std::pair<std::string, double> >resurest;

	search_face_object(*pface, personslist, resurest);

	if (resurest.size()>0)
	{
		// find the min dis
		pvPersonDist->resurest.clear();
		tDis2Gallery tTmp;
		for(int i = 0; i<resurest.size(); i++)
		{
			tTmp.fdis = resurest[i].second;
			tTmp.nLabel = string2int(resurest[i].first.c_str());
			tTmp.nId = i;
			pvPersonDist->resurest.push_back(tTmp);
		}
		pvPersonDist->bSignIded = false;

#if 1
		for (int i = 0; i<resurest.size(); i++)
		{
			printf("name %s, conf: %f\n", resurest[i].first.c_str(), float(resurest[i].second));
		}
#endif
		return true;
	}
	return false;
}
*/

/*
int CFaceUit::facepredict(int anLabels[])
{
	int nFlag = -1;
	if (personslist.size()<1)return -1;

	float aFeats[MAX_NUM_FEATS];
	
	//tPersonShoot.
	if (1)
	{
		anLabels[0] = g_NumFaceDet;
#if 1
		faimg.clear();
#endif

#if 1
		if(getface(faimg, 1)) {
#else
		if (1) {
#endif
			for (int ii = 0; ii < g_NumFaceDet; ii++)
			{
				//printf("ii is %d", ii);
				CFace face;
				if (pcCalFeatExtracrt->extractDescriptors(faimg, ii))
				{
					pcCalFeatExtracrt->get_feats(aFeats);
					face.copy_feat(aFeats, MAX_NUM_FEATS);
					cv::Mat cStandardImg = pcCalFeatExtracrt->get_img();
					face.set_img(cStandardImg);

					std::string result_name;
					double dis;
					if (predict(&face, result_name, dis)){
						anLabels[ii+1] = string2int(result_name.c_str());
						nFlag = +1;
					}
					else
					{
						anLabels[ii+1] = -1;
					}
				}
			}
		}
	}
	return nFlag;
}

int CFaceUit::facepredict4mul(int anLabels[])
{
	int nFlag = -1;
#if 1
	if (personslist.size()<1)return -1;
#endif

	float aFeats[MAX_NUM_FEATS];

	memset(anLabels, -1, sizeof(int)*g_NumFaceDet);
	anLabels[0] = g_NumFaceDet;
	std::vector<tPersonCmp*> vPersonDist;
#if 1
	faimg.clear();
#endif

#if 1
	if (getface(faimg, 1)) 
#else
	if (1) 
#endif
	{
		for (int ii = 0; ii < g_NumFaceDet; ii++)
		{
			//printf("ii is %d", ii);
			CFace face;
			if (pcCalFeatExtracrt->extractDescriptors(faimg, ii))
			//if (pcCalFeatExtracrt->extractDescriptors(faimg, 1))
			{
#if 1
				pcCalFeatExtracrt->get_feats(aFeats);
				face.copy_feat(aFeats, MAX_NUM_FEATS);
				cv::Mat cStandardImg = pcCalFeatExtracrt->get_img();
				face.set_img(cStandardImg);

				tPersonCmp *ptPersonCmp = new tPersonCmp;
				ptPersonCmp->nOwnId = ii;
				if (predict4mul(&face, ptPersonCmp))
				{
					vPersonDist.push_back(ptPersonCmp);
				}
#endif
			}
		}
#if 1
		nFlag = abilixMulPersonPredict(vPersonDist, g_thresh);
		for (int ii = 0; ii < int(vPersonDist.size()); ii++)
		{
			if (vPersonDist[ii]->bSignIded == true)
			{
				int idx = vPersonDist[ii]->nOwnId;
				if (vPersonDist[ii]->bSignIded == true)
				{
					anLabels[idx + 1] = vPersonDist[ii]->nLabel;
				}
				else
				{
					anLabels[idx + 1] = -1;
				}
			}
		}
#endif
	}
#if 1
	for (int ii = 0; ii < (int)vPersonDist.size(); ii++)
	{
		if (NULL != vPersonDist[ii])
		{
			delete vPersonDist[ii];
			vPersonDist[ii] = NULL;
		}
	}
#endif
	return nFlag;
}
*/

static int filterOutArray(float *pLabelArray)
{
	std::list<float>::iterator it;
	int nCnt;
	float afLabelArray[N_FACE_SHAPE];
	int anCntArray[N_FACE_SHAPE];

	memcpy(afLabelArray, pLabelArray, sizeof(float) * N_FACE_SHAPE);

	if (gEllipseList.size() < MAX_LIST_SIZE)
	{
		gEllipseList.push_back(pLabelArray[0]);
	}
	else if (gEllipseList.size() >= MAX_LIST_SIZE)
	{
		gEllipseList.push_back(pLabelArray[0]);
		gEllipseList.pop_front();
	}

	if (gRoundList.size() < MAX_LIST_SIZE)
	{
		gRoundList.push_back(pLabelArray[1]);
	}
	else if (gRoundList.size() >= MAX_LIST_SIZE)
	{
		gRoundList.push_back(pLabelArray[1]);
		gRoundList.pop_front();
	}

	if (gSquareList.size() < MAX_LIST_SIZE)
	{
		gSquareList.push_back(pLabelArray[2]);
	}
	else if (gSquareList.size() >= MAX_LIST_SIZE)
	{
		gSquareList.push_back(pLabelArray[2]);
		gSquareList.pop_front();
	}

	if (gTriangleList.size() < MAX_LIST_SIZE)
	{
		gTriangleList.push_back(pLabelArray[3]);
	}
	else if (gTriangleList.size() >= MAX_LIST_SIZE)
	{
		gTriangleList.push_back(pLabelArray[3]);
		gTriangleList.pop_front();
	}

	nCnt = 0;
	for (it = gEllipseList.begin(); it != gEllipseList.end(); it++)
	{
		nCnt += (*it > 1e-5) ? 1 : 0;
	}
	anCntArray[0] = nCnt;

	nCnt = 0;
	for (it = gRoundList.begin(); it != gRoundList.end(); it++)
	{
		nCnt += (*it > 1e-5) ? 1 : 0;
	}
	anCntArray[1] = nCnt;

	nCnt = 0;
	for (it = gSquareList.begin(); it != gSquareList.end(); it++)
	{
		nCnt += (*it > 1e-5) ? 1 : 0;
	}
	anCntArray[2] = nCnt;

	nCnt = 0;
	for (it = gTriangleList.begin(); it != gTriangleList.end(); it++)
	{
		nCnt += (*it > 1e-5) ? 1 : 0;
	}
	anCntArray[3] = nCnt;

	nCnt = 0;
	for (int ii = 0; ii < N_FACE_SHAPE; ii++)
	{
#if 1
		printf("pLabelArray[%d]: %f, ", ii, anCntArray[ii]);
#endif
		pLabelArray[ii] = (anCntArray[ii] >= 2) ? 1 : 0;
		nCnt += (pLabelArray[ii] > 1e-5) ? 1 : 0;
	}
#if 1
	printf("\n");
#endif
	if (nCnt == 0)
	{
		if (gEllipseList.size() == 1) {
			for (int ii = 0; ii < N_FACE_SHAPE; ii++) {
				pLabelArray[ii] = (afLabelArray[ii] > 1e-5) ? 1 : 0;
			}
		}
		else if (gEllipseList.size() > 1)
		{
			for (int ii = 0; ii < N_FACE_SHAPE; ii++) {
				if (anCntArray[ii] > 1)
					pLabelArray[ii] = (afLabelArray[ii] > 1e-5) ? 1 : 0;
			}
		}
	}
	return 0;
}

int CFaceUit::facepredict4mul(float anLabels[])
{
	int nFlag = -1;
	float aFeats[MAX_NUM_FEATS];
	int nRet = -1;
	float *pTemp = NULL;
	float *pTemp0 = NULL;
	int nStatus = -1;

	memset(anLabels, -1, sizeof(float)*g_NumFaceDet*OUTPUT_ARRAY_STEP);
	if (g_GetOneFace == 1)
		anLabels[0] = 1;
	else
		anLabels[0] = g_NumFaceDet;


#if 1
	faimg.clear();
#endif

#if 1
	if (getface(faimg, 1))
#else
	if (1)
#endif
	{
		for (int ii = 0; ii < anLabels[0]; ii++)
		{
			//printf("ii is %d", ii);
			CFace face;
			if (pcCalFeatExtracrt->extractDescriptors(faimg, ii))
			{
#if 1
				pcCalFeatExtracrt->get_feats(aFeats);
				//Get label for gender;
				float fMaxProb = -999.9f;
				int nOutLabel = -1;
				int nCnt = 0;

				pTemp0 = anLabels + ii*OUTPUT_ARRAY_STEP + 1;
				pTemp = anLabels + ii*OUTPUT_ARRAY_STEP + 1 + 1;
				//1. transform the probability of each type to binary label through thresholding;
				for (int jj = 0; jj < N_FACE_SHAPE; jj++)
				{
					printf("aFeats %f\n", aFeats[jj]);

					pTemp[jj] = (aFeats[jj]>g_thresholds[jj]) ? 1 : 0;
					pTemp[jj + 4] = aFeats[jj];
					nCnt += (pTemp[jj] > 0) ? 1 : 0;
				}
				pTemp[8] = aFeats[5];
				pTemp[9] = aFeats[6];

#if 0
				printf("alpha: %f\n", anLabels[ii * 7 + 1 + 4 + 1]);
				printf("beta: %f\n", anLabels[ii * 7 + 1 + 5 + 1]);
#endif
				//2. push the binary labels to the lists;
				nStatus = filterOutArray(pTemp);
				if (0 != nStatus)
				{
					nFlag = -3;
				}
				// 3.if all labels are zeros, then output -1;
				nCnt = 0;
				for (int jj = 0; jj < N_FACE_SHAPE; jj++)
				{
					nCnt += (pTemp[jj] > 0) ? 1 : 0;;
				}
				nOutLabel = (nCnt > 0) ? nCnt : -1;

				// 4.if the pose estimation is outside the range, then output -2.
#if 1

				if ((pTemp[8] < 30) && (pTemp[9] < 30))
				{

					pTemp0[0] = (nOutLabel > 0) ? 0 : -1;
					nFlag = 0;
				}
				else
				{

					pTemp0[0] = -2;
					nFlag = -2;
				}
#else
				anLabels[ii * 7 + 1] = nOutLabel;
#endif

#if 0
				printf("noutlabel: %d\n", nOutLabel);
				printf("anLabels[ii*7 + 1]: %f\n", anLabels[ii * 7 + 1]);
#endif

#endif
			}
		}
	}

	return nFlag;
}


int CFaceUit::getpersonList(std::vector<int> &personList)
{
	personList.clear();
	if (personslist.size()<1)return -1;

	for (int ii = 0; ii < personslist.size(); ii++)
	{
		string sNameTmp = personslist[ii]->get_name();
		int nNameTmp = string2int(sNameTmp.c_str());
		personList.push_back(nNameTmp);
	}
	return 0;
}
