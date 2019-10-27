#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

//Mnist����
typedef struct _MnistData
{
	double** pData;		//ͼ������
	double** pLable;	//��ǩ����

	int nWidth;			//ͼ����
	int nHeight;		//ͼ��߶�
	int nNumber;		//ͼ������
	int nClassNumber;	//�������

	_MnistData():pData(nullptr),pLable(nullptr),nWidth(0),nHeight(0),nNumber(0),nClassNumber(10) { }

}MnistData,*PMnistData;

//��С�ֽ�ת��
int ReversalInt(int nValue);

//��ȡMnistͼ������
bool ReadMnistData(MnistData& stMnist,const std::string& strPath,int nPadding = 2);

//��ȡMnist��ǩ����
bool ReadMnistLable(MnistData& stMnist,const std::string& strPath);

//�ͷ�Mnist����
bool ReleaseMnistData(MnistData& stMnist);





















