#pragma once

#include <chrono>
#include "Mnist.h"

class Time
{
private:
	using SystemTime = std::chrono::high_resolution_clock;
	//��ʼʱ��
	std::chrono::time_point<SystemTime> m_cBeginTime;

	//����ʱ��
	std::chrono::time_point<SystemTime> m_cEndTime;

public:
	Time() :m_cBeginTime(SystemTime::now()) {};
	~Time() {};

public:
	//����ʱ��
	void ReSetTime() { m_cBeginTime = SystemTime::now(); };

	//��ȡʱ���
	double GetTimeCount()
	{
		m_cEndTime = SystemTime::now();
		long long lTime = std::chrono::duration_cast<std::chrono::milliseconds>(m_cEndTime - m_cBeginTime).count();
		m_cBeginTime = m_cEndTime;
		return static_cast<double>(lTime) / 1000.0 / 60.0;
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//�����
typedef struct _Kernel
{
	double* pWeight;		//Ȩ��
	double* pDw;			//δ֪
	void Release() { pWeight = pDw = nullptr; }
	_Kernel() :pWeight(nullptr), pDw(nullptr) {}
}Kernel,*PKernel;

//ͼ������
typedef struct _Map
{
	double* pData;			//�������
	double* pError;			//�������
	double dBias;			//ƫ������
	double dDb;				//�ܴ�����
	void Release() { pData = pError = nullptr; dBias = dDb = 0.0; }
	_Map():pData(nullptr),pError(nullptr),dBias(0),dDb(0) {}
}Map,*PMap;

//�����
typedef struct _Layer
{
	int nMapWidth;			//ͼ����
	int nMapHeight;			//ͼ��߶�
	int nMapCount;			//ͼ������
	PMap pMap;				//ͼ��ָ��

	int nKernelWidth;		//����˿��
	int nKernelHeight;		//����˸߶�
	int nKernelCount;		//���������
	PKernel pKernel;		//�����ָ��

	double* pMapCommon;		//δ֪
	void Release() 
	{
		nMapWidth = nMapHeight = nMapCount = 0;
		nKernelWidth = nKernelHeight = nKernelCount = 0;
		pMapCommon = nullptr;
	}
	_Layer() :nMapWidth(0),nMapHeight(0),nMapCount(0),pMap(nullptr),
	nKernelWidth(0),nKernelHeight(0),nKernelCount(0),pKernel(nullptr),pMapCommon(nullptr){}
}Layer,*PLayer;

//Mnist���������
typedef struct _MnistNet
{
	Layer stInputLayer_0;	//�����	
	Layer stConvLayer_1;	//�����
	Layer stPoolLayer_2;	//�ػ���
	Layer stConvLayer_3;	//�����
	Layer stPoolLayer_4;	//�ػ���
	Layer stConvLayer_5;	//�����
	Layer stOutputLayer_6;	//�����

}MnistNet,*PMnistNet;

//���ӱ�
#define Y true
#define N false
static bool NetConnectTable[] =
{
	Y, N, N, N, Y, Y, Y, N, N, Y, Y, Y, Y, N, Y, Y,
	Y, Y, N, N, N, Y, Y, Y, N, N, Y, Y, Y, Y, N, Y,
	Y, Y, Y, N, N, N, Y, Y, Y, N, N, Y, N, Y, Y, Y,
	N, Y, Y, Y, N, N, Y, Y, Y, Y, N, N, Y, N, Y, Y,
	N, N, Y, Y, Y, N, N, Y, Y, Y, Y, N, Y, Y, N, Y,
	N, N, N, Y, Y, Y, N, N, Y, Y, Y, Y, N, Y, Y, Y
};
#undef Y
#undef N

//Double����Ч��Χ
inline bool IsValidDouble(double dValue) { return (dValue <= DBL_MAX && dValue >= -DBL_MAX); };

//��ʼ�������
bool InitializeKernel(double* pWeight,	//Ȩ�ص�ַ
	int nKernelSize,					//����˴�С
	double dWeightBase);				//Ȩ�ػ�׼

//��ʼ�������
bool InitializeLayer(Layer& stLayer,	//��ǰ��
	int nPreviousLayerMapNumber,		//��һ��ͼ������
	int nOutputMapNumber,				//��ǰ�����ͼ������
	int nKernelWidth,					//����˿��
	int nKernelHeight,					//����˸߶�
	int nInputMapWidth,					//����ͼ����
	int nInputMapHeight,				//����ͼ��߶�
	bool bIsPooling = false);			//�Ƿ�ػ�

//��ʼ������
bool InitializeMnistNet(MnistNet& stMnistNet,	//�ṹָ��
	int nWidth,									//����ͼ����
	int nHeight,								//����ͼ��߶�
	int nClassNumber);							//Ԥ���������

//��ʼѵ��ģ��
bool TrainModel(MnistNet& stMnistNet,			//����ṹ
	MnistData& stMnistTrain,					//ѵ��������
	MnistData& stMnistTest,						//���Լ�����
	double dLearningRate,						//ѧϰ��
	int nBatchSize,								//��������
	int nEpoch = 5);							//��������

//����Ȩ��
bool ResetWeight(MnistNet& stMnistNet);			//����ṹ

//���ò�
bool ResetLayer(Layer& stLayer);				//��ṹ

//����Ȩ��
bool UpdateWeight(MnistNet& stMnistNet,			//����ṹ
	double dLearnigrate,						//ѧϰ��
	int nBatchSize);							//��������

//���²�
bool UpdateLayer(Layer& stLayer,				//��ṹ
	double dLearningRate,						//ѧϰ��
	int nBatchSize);							//��������			

//�ݶ��½��㷨
double GradientDescent(double dWeight,			//��ǰȨ��
	double dWd,									//δ֪
	double dLearningRate,						//ѧϰ��
	double dLambda);							//δ֪

//ǰ�򴫲�
bool ForwardPropagation(MnistNet& stMnistNet);	//����ṹ

//���򴫲�
bool BackwardPropagation(MnistNet& stMnistNet,	//����ṹ
	double* pLabelData);						//��ǩ����

//������ǰ�򴫲�
bool ForwardToConvolution(Layer& stPreviouLayer,//ǰһ��
	Layer& stCurrentLayer,						//��ǰ��
	const bool* pConnectTable = nullptr);		//���ӱ�ָ��

//�ػ����ǰ�򴫲�,�����õ������ػ�
bool ForwardToPooling(Layer& stPreviouLayer,	//ǰһ��
	Layer& stCurrentLayer);						//��ǰ��

//ȫ���Ӳ��ǰ�򴫲�
bool ForwardToFullConnect(Layer& stPreviouLayer,//ǰһ��
	Layer& stCurrentLayer);						//��ǰ��

//��Ч���
bool ValidConvolution(double* pInputData,		//����ͼ������
	int nInputWidth,							//����ͼ����
	int nInputHeight,							//����ͼ��߶�
	double* pKernelData,						//���������
	int nKernelWidth,							//����˿��
	int nKernelHeight,							//����˸߶�
	double* pOutputData,						//���ͼ������
	int nOutputWidth,							//���ͼ����
	int nOuputHeight);							//���ͼ��߶�

//�����Tanh
double ActivationTanh(double dValue);

//�����Tanh�ĵ���
double DerivativeTanh(double dValue);

//�����Relu
double ActivationRelu(double dValue);

//�����Relu�ĵ���
double DerivativeRelu(double dValue);

//�����Sigmoid
double ActivationSigmoid(double dValue);

//�����Sigmoid�ĵ���
double DerivativeSigmoid(double dValue);

//ȫ���Ӳ�ķ��򴫲�
bool BackwardToFullConnect(Layer& stCurrentLayer,	//��ǰ��
	Layer& stPreviouLayer);							//��һ��

//�����ķ��򴫲�
bool BackwardToConvolution(Layer& stCurrentLayer,	//��ǰ��
	Layer& stPreviouLayer,							//��һ��
	const bool* pConnectTable = nullptr);			//���ӱ�

//�ػ���ķ��򴫲�
bool BackwardToPooling(Layer& stCurrentLayer,		//��ǰ��
	Layer& stPreviouLayer);							//��һ��

//ģ��Ԥ��
bool Predicts(MnistNet& stMnistNet,					//����ṹ
	MnistData& stMnistData);						//Mnist����

//��ȡ���ֵ����
int GetOutputIndex(Layer& stOutputLayer);			//�����

//��ȡʵ��ֵ����
int GetActualIndex(double* pLabel,					//��ǩ����
	int nClassNumber);								//�������

//�ͷ�����ṹ
bool ReleaseMnistNet(MnistNet& stMnistNet);			//����ṹ

//�ͷŲ�ṹ
bool ReleaseLayer(Layer& stLayer);					//��ṹ





