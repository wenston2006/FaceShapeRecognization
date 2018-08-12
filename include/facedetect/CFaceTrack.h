#ifndef _CFACE_TRACK_H_
#define _CFACE_TRACK_H_

#include <string>
#include "opencv2/opencv.hpp"

#define ABILIX_FACE_POSE_PI							3.1415926
#define ABILIX_MOUTH_LEFT_CORNER_ID					48
#define ABILIX_MOUTH_RIGHT_CORNER_ID				54
#define ABILIX_NOSE_ID								30
#define ABILIX_FACE_RATIO_AT_NOSE_INVERSE			1.2632

#ifdef FACEDETECT_EXPORTS
#define FACEDETECT_API __declspec(dllexport)
#else
#define FACEDETECT_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

	FACEDETECT_API int facepose2init(int w, int h, std::string dbdir);

	FACEDETECT_API int facepose2process(unsigned char* Image, int distpos[100]);

	FACEDETECT_API int facepose2detect(char* Image, float *facerect);

	FACEDETECT_API int facepose2destroy();

#ifdef __cplusplus
}
#endif

#endif
