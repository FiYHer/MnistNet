#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

//Mnist数据
typedef struct _MnistData
{
	double** pData;		//图像数据
	double** pLable;	//标签数据

	int nWidth;			//图像宽度
	int nHeight;		//图像高度
	int nNumber;		//图像数量
	int nClassNumber;	//类别数量

	_MnistData():pData(nullptr),pLable(nullptr),nWidth(0),nHeight(0),nNumber(0),nClassNumber(10) { }

}MnistData,*PMnistData;

//大小字节转换
int ReversalInt(int nValue);

//读取Mnist图像数据
bool ReadMnistData(MnistData& stMnist,const std::string& strPath,int nPadding = 2);

//读取Mnist标签数据
bool ReadMnistLable(MnistData& stMnist,const std::string& strPath);

//释放Mnist数据
bool ReleaseMnistData(MnistData& stMnist);





















