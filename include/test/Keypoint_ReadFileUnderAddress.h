#ifndef _READ_FILE_UNDER_ADDRESS_
#define _READ_FILE_UNDER_ADDRESS_

#ifdef WIN32
#include <io.h> //use _findfirst() and findnext() function
#include <shlwapi.h>
#include <string>
#include <vector>
#pragma comment(lib, "shlwapi.lib")
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif // WIN32
/*=========================================================================
函数名   ：Recurse
功能     ：遍历pstr所指向的目录，查找pstrName所指向的文件，并存储在cstraFile中
算法实现 ：无
参数说明 ：pstr         指向要查找的目录
pstrName     要查找的文件名
vecFileName  存储查找结果
u32OnlyFlag  每个文件夹下找到一个就返回标志
返回值说明：无
其他说明 ：
==========================================================================*/

using namespace std;

typedef unsigned int       u32;

#ifdef WIN32
void Recurse(string &pstr,string &pstrName, std::vector<string> &vecFileName,u32 u32OnlyFlag)
{
	struct _finddata_t t_file; //定义结构体变量
	long handle;
	string strFind;
	string strFileName;

	strFind = pstr + "\\*";

	handle = _findfirst(strFind.c_str(),&t_file);//查找所有文件

	// 表示当前目录为空
	if( -1L == handle )
	{
		return;
	}
	else
	{
		//是目录
		if( t_file.attrib & _A_SUBDIR ) 
		{
			//文件名不是'.'或'..'时
			if( t_file.name[0] != '.') 
			{
				//遍历该目录
				strFind = pstr + "\\" + t_file.name;

				Recurse(strFind,pstrName,vecFileName,u32OnlyFlag);
			}
		}
		// 是文件
		else
		{
			string strFilePath = pstr + "\\" + t_file.name;

			strFileName = t_file.name;

			if( strFileName.find(pstrName,0) != -1 )
			{
				vecFileName.push_back(strFilePath);

				if( u32OnlyFlag )
				{
					_findclose(handle);
					return;
				}
			}
		}

		// 继续对当前目录中的下一个子目录或文件进行与上面同样的查找
		while(!(_findnext(handle,&t_file)))
		{
			if( t_file.attrib & _A_SUBDIR ) //是目录
			{
				if( t_file.name[0] != '.' ) //文件名不是'.'或'..'时
				{
					//遍历该目录
					strFind = pstr + "\\" + t_file.name;

					Recurse(strFind,pstrName,vecFileName,u32OnlyFlag);
				}
			}
			else // 是文件
			{
				string strFilePath = pstr + "\\" + t_file.name;

				strFileName = t_file.name;

				if( strFileName.find(pstrName,0) != -1 )
				{
					vecFileName.push_back(strFilePath);

					if( u32OnlyFlag )
					{
						_findclose(handle);
						return;
					}
				}
			}
		}
		_findclose(handle);
	}
}
#else
static void Recurse(string &pstr, string &pstrName, std::vector<string> &vecFileName, u32 u32OnlyFlag)
{
	struct dirent* ent = NULL;
	DIR *pDir = NULL;
	string strFileName;
	string strFind;

	pDir=opendir(pstr.c_str());
	if(pDir == NULL)   
	{        
		//printf("error opendir %s!!!\n",pstr.c_str());  
		return;   
	}
	while (NULL != (ent=readdir(pDir)))
	{
		//printf("ent name：%s\n",ent->d_name);

		string strFilePath = pstr + "/" + ent->d_name;
		strFileName = ent->d_name;
		//if(strFileName.find(pstrName, 0) != -1)
		//strcmp(s1,s2))
        if(!strcmp(ent->d_name,pstrName.c_str()))
		{
			vecFileName.push_back(strFilePath);
		}

		/*if (ent->d_reclen==24)
		{
			printf("ent type %d\n", ent->d_type);
			if (ent->d_type==8)
			{
				printf("普通文件:%s\n", ent->d_name);
				string strFilePath = pstr + "//" + ent->d_name;
				strFileName = ent->d_name;

				if(strFileName.find(pstrName, 0) != -1)
				{
					vecFileName.push_back(strFilePath);
				}
			}
			else
			{
				printf("子目录：%s\n",ent->d_name);
				//List(ent->d_name);
				if(ent->d_name[0] != '.')
				{
					strFind = pstr + "//" + ent->d_name;
					Recurse(strFind, pstrName, vecFileName, 1);
				}
				printf("返回%s\n",ent->d_name);
			}
		}*/
	}
	closedir(pDir);

}
#endif
#endif
