#ifndef _HAAR_H_
#define _HAAR_H_

#define __int64 long

namespace fadt{

#define MAXROWS   1600   
#define MAXCOLS   1600	
#define MAXSEQS    25
	
typedef unsigned char	(*ImgPtr);
typedef unsigned char	(*Mat8Ptr);
typedef int				(*Mat32Ptr);
typedef __int64			(*Mat64Ptr);
//typedef long long		(*Mat64Ptr);

/*****************���鼯���ݽṹ*******************************/
#define MAXPTREENODES 100
typedef struct PTreeNode
{
    struct PTreeNode* parent;
    char* element;
    int rank;
}PTreeNode;

/************************����ͼ����***************************/
typedef int sumtype;
typedef __int64 sqsumtype;
//typedef long long sqsumtype;

/************************************************************/
typedef struct Rect
{
	   int x;
	   int y;
	   int width;
	   int height;
}Rect;

typedef struct
{
    int width;
    int height;
}Size;

typedef struct Image
{
 	ImgPtr  imgPtr;
	int rows;
	int cols;
}Image;

typedef struct Mat8
{
	Mat8Ptr  mat8Ptr;
	int rows;
	int cols;
}Mat8;
typedef struct Mat32
{
	Mat32Ptr  mat32Ptr;
	int rows;
	int cols;
}Mat32;

typedef struct Mat64
{
	Mat64Ptr  mat64Ptr;
	int rows;
	int cols;
}Mat64;

typedef struct Sequence
{
	int       total; 
	Rect	  rectQueue[MAXSEQS];
	int		  neighbors[MAXSEQS];
	int		  tail;
}Sequence;


//Haar����������
#define CV_HAAR_FEATURE_MAX  3    

/*************HidHaar to Caculation Feature***********************************/
typedef struct HidHaarFeature
{
    struct
    {
        sumtype *p0, *p1, *p2, *p3;
        int weight;
    }
    rect[CV_HAAR_FEATURE_MAX];
}HidHaarFeature;


typedef struct HidHaarTreeNode
{
    HidHaarFeature feature;
    int threshold;
    int left;
    int right;
}HidHaarTreeNode;


typedef struct HidHaarClassifier
{
    int count;
    //CvHaarFeature* orig_feature;

    HidHaarTreeNode* node;
    int* alpha;
	//HidHaarTreeNode node[MAXTREENODE];
    //int alpha[MAXALPHA];
}HidHaarClassifier;

typedef struct HidHaarStageClassifier
{
    int  count;
    int threshold;
    HidHaarClassifier* classifier;
//	HidHaarClassifier classifier[MAXCLASSIFER];
    int two_rects;
    
    struct HidHaarStageClassifier* next;
    struct HidHaarStageClassifier* child;
    struct HidHaarStageClassifier* parent;
}HidHaarStageClassifier;


typedef struct HidHaarClassifierCascade
{
    int  count;
    int  is_stump_based;
    int  has_tilted_features;
    int  is_tree;
    int window_area;
    Mat32* sum;
	Mat64* sqsum;
    HidHaarStageClassifier* stage_classifier;
	//HidHaarStageClassifier stage_classifier[MAXSTAGES];
    sqsumtype *pq0, *pq1, *pq2, *pq3;
    sumtype *p0, *p1, *p2, *p3;
	
    void** ipp_stages;
}HidHaarClassifierCascade;



/******************Haar Cascade*****************************************/
typedef struct HaarFeature
{
    int  tilted;
    struct
    {
        Rect r;
        int weight;
    } rect[CV_HAAR_FEATURE_MAX];
}HaarFeature;

typedef struct HaarClassifier
{
    int count;
    HaarFeature* haar_feature;
    int* threshold;
    int* left;
    int* right;
    int* alpha;
}HaarClassifier;

typedef struct HaarStageClassifier
{
    int  count;
    int threshold;
    HaarClassifier* classifier;
	
    int next;
    int child;
    int parent;
}HaarStageClassifier;


typedef struct HaarClassifierCascade
{
    int  flags;
    int  count;
    Size orig_window_size;
    Size real_window_size;
    int scale32x;
    HaarStageClassifier* stage_classifier;
    HidHaarClassifierCascade* hid_cascade;
}HaarClassifierCascade;

typedef struct Point
{
	int x;
	int y;
}Point;

/******************ȫ�ֱ���****************************************/
//8bits*3 cell 
extern unsigned char  ImgRGBPool8[MAXROWS][MAXCOLS];

extern Sequence result_seq;//�������������������

/********************ȫ�ֺ���******************************************/
extern void ReadFaceCascade();
extern void HaarDetectObjects(const Image* _img,const int scale_factor32x,int min_neighbors,const Size minSize);

void DownSample(Image* pImage, unsigned factor);

}
#endif