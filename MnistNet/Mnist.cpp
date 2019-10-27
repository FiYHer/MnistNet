#include "Mnist.h"
using namespace std;

int ReversalInt(int nValue)
{
	unsigned char cTemp1 = nValue & 255;
	unsigned char cTemp2 = (nValue >> 8) & 255;
	unsigned char cTemp3 = (nValue >> 16) & 255;
	unsigned char cTemp4 = (nValue >> 24) & 255;
	int nData = static_cast<int>(cTemp1) << 24;
	nData += static_cast<int>(cTemp2) << 16;
	nData += static_cast<int>(cTemp3) << 8;
	return nData + cTemp4;
}

bool ReadMnistData(MnistData& stMnist, 
	const string& strPath,
	int nPadding)
{
	if (strPath.empty())return false;

	//二进制方式读取
	fstream cMnistFile(strPath,fstream::in | fstream::binary);
	if (!cMnistFile.is_open())return false;

	int nMagic = 0, nNumber = 0, nWidth = 0, nHeight = 0;

	cMnistFile.read(reinterpret_cast<char*>(&nMagic), sizeof(nMagic));
	cMnistFile.read(reinterpret_cast<char*>(&nNumber),sizeof(nNumber));
	cMnistFile.read(reinterpret_cast<char*>(&nWidth), sizeof(nWidth));
	cMnistFile.read(reinterpret_cast<char*>(&nHeight), sizeof(nHeight));

	nMagic = ReversalInt(nMagic);
	if (nMagic != 2051)return false;

	nNumber = ReversalInt(nNumber);
	nWidth = ReversalInt(nWidth);
	nHeight = ReversalInt(nHeight);

	stMnist.nNumber = nNumber;

	//这里加上填充，乘以2是为了上下和左右都加上填充
	stMnist.nWidth = nWidth + nPadding * 2;
	stMnist.nHeight = nHeight + nPadding * 2;

	double dScaleMax = 1.0, dScaleMin = -1.0;

	int nSize = stMnist.nWidth * stMnist.nHeight;

	stMnist.pData = new double*[nNumber];

	//图像数量
	for (int i = 0; i < nNumber; i++)
	{
		//初始化为-1.0
		stMnist.pData[i] = new double[nSize];
		for (int j = 0; j < nSize; j++)stMnist.pData[i][j] = -1.0;

		for (int j = 0; j < nHeight; j++)
		{
			for (int k = 0; k < nWidth; k++)
			{
				unsigned char cTemp;
				cMnistFile.read(reinterpret_cast<char*>(&cTemp), sizeof(cTemp));
				
				double dTemp = (static_cast<double>(cTemp) / 255.0) * (dScaleMax - dScaleMin) + dScaleMin;
				stMnist.pData[i][(j + nPadding) * stMnist.nWidth + k + nPadding] = dTemp;
			}
		}
	}

	cMnistFile.close();

	return true;
}

bool ReadMnistLable(MnistData& stMnist, 
	const std::string& strPath)
{
	if (strPath.empty()) return false;

	fstream cMnistFile(strPath,fstream::in | fstream::binary);
	if (!cMnistFile.is_open()) return false;

	int nMagic = 0, nNumber = 0;

	cMnistFile.read(reinterpret_cast<char*>(&nMagic),sizeof(nMagic));
	cMnistFile.read(reinterpret_cast<char*>(&nNumber),sizeof(nNumber));

	nMagic = ReversalInt(nMagic);
	nNumber = ReversalInt(nNumber);
	if (nMagic != 2049 || nNumber != stMnist.nNumber)return false;

	char cIndex = 0;

	stMnist.pLable = new double*[nNumber];
	
	//正确标签为0.8，其它为-0.8
	for (int k = 0; k < nNumber; k++)
	{
		stMnist.pLable[k] = new double[stMnist.nClassNumber];
		for (int j = 0; j < stMnist.nClassNumber; j++)stMnist.pLable[k][j] = -0.8;

		cMnistFile.read(reinterpret_cast<char*>(&cIndex), sizeof(cIndex));
		stMnist.pLable[k][cIndex] = 0.8;
	}

	cMnistFile.close();

	return true;
}

bool ReleaseMnistData(MnistData& stMnist)
{
	//释放图像数据
	if (stMnist.pData)
	{
		for (int i = 0; i < stMnist.nNumber; i++)
		{
			if (stMnist.pData[i])
				delete[] stMnist.pData[i];
		}
		delete[] stMnist.pData;
		stMnist.pData = nullptr;
	}

	//释放标签数据
	if (stMnist.pLable)
	{
		for (int i = 0; i < stMnist.nClassNumber; i++)
		{
			if (stMnist.pLable[i])
				delete[] stMnist.pLable[i];
		}
		delete[] stMnist.pLable;
		stMnist.pLable = nullptr;
	}

	return true;
}
