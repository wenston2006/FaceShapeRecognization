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
������   ��Recurse
����     ������pstr��ָ���Ŀ¼������pstrName��ָ����ļ������洢��cstraFile��
�㷨ʵ�� ����
����˵�� ��pstr         ָ��Ҫ���ҵ�Ŀ¼
pstrName     Ҫ���ҵ��ļ���
vecFileName  �洢���ҽ��
u32OnlyFlag  ÿ���ļ������ҵ�һ���ͷ��ر�־
����ֵ˵������
����˵�� ��
==========================================================================*/

using namespace std;

typedef unsigned int       u32;

#ifdef WIN32
void Recurse(string &pstr,string &pstrName, std::vector<string> &vecFileName,u32 u32OnlyFlag)
{
	struct _finddata_t t_file; //����ṹ�����
	long handle;
	string strFind;
	string strFileName;

	strFind = pstr + "\\*";

	handle = _findfirst(strFind.c_str(),&t_file);//���������ļ�

	// ��ʾ��ǰĿ¼Ϊ��
	if( -1L == handle )
	{
		return;
	}
	else
	{
		//��Ŀ¼
		if( t_file.attrib & _A_SUBDIR ) 
		{
			//�ļ�������'.'��'..'ʱ
			if( t_file.name[0] != '.') 
			{
				//������Ŀ¼
				strFind = pstr + "\\" + t_file.name;

				Recurse(strFind,pstrName,vecFileName,u32OnlyFlag);
			}
		}
		// ���ļ�
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

		// �����Ե�ǰĿ¼�е���һ����Ŀ¼���ļ�����������ͬ���Ĳ���
		while(!(_findnext(handle,&t_file)))
		{
			if( t_file.attrib & _A_SUBDIR ) //��Ŀ¼
			{
				if( t_file.name[0] != '.' ) //�ļ�������'.'��'..'ʱ
				{
					//������Ŀ¼
					strFind = pstr + "\\" + t_file.name;

					Recurse(strFind,pstrName,vecFileName,u32OnlyFlag);
				}
			}
			else // ���ļ�
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
		//printf("ent name��%s\n",ent->d_name);

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
				printf("��ͨ�ļ�:%s\n", ent->d_name);
				string strFilePath = pstr + "//" + ent->d_name;
				strFileName = ent->d_name;

				if(strFileName.find(pstrName, 0) != -1)
				{
					vecFileName.push_back(strFilePath);
				}
			}
			else
			{
				printf("��Ŀ¼��%s\n",ent->d_name);
				//List(ent->d_name);
				if(ent->d_name[0] != '.')
				{
					strFind = pstr + "//" + ent->d_name;
					Recurse(strFind, pstrName, vecFileName, 1);
				}
				printf("����%s\n",ent->d_name);
			}
		}*/
	}
	closedir(pDir);

}
#endif
#endif
