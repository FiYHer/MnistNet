#pragma once

#include <chrono>
#include "Mnist.h"

class Time
{
private:
	using SystemTime = std::chrono::high_resolution_clock;
	//开始时间
	std::chrono::time_point<SystemTime> m_cBeginTime;

	//结束时间
	std::chrono::time_point<SystemTime> m_cEndTime;

public:
	Time() :m_cBeginTime(SystemTime::now()) {};
	~Time() {};

public:
	//重置时间
	void ReSetTime() { m_cBeginTime = SystemTime::now(); };

	//获取时间差
	double GetTimeCount()
	{
		m_cEndTime = SystemTime::now();
		long long lTime = std::chrono::duration_cast<std::chrono::milliseconds>(m_cEndTime - m_cBeginTime).count();
		m_cBeginTime = m_cEndTime;
		return static_cast<double>(lTime) / 1000.0 / 60.0;
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//卷积核
typedef struct _Kernel
{
	double* pWeight;		//权重
	double* pDw;			//未知
	void Release() { pWeight = pDw = nullptr; }
	_Kernel() :pWeight(nullptr), pDw(nullptr) {}
}Kernel,*PKernel;

//图像数据
typedef struct _Map
{
	double* pData;			//输出数据
	double* pError;			//误差数据
	double dBias;			//偏置数据
	double dDb;				//总错误率
	void Release() { pData = pError = nullptr; dBias = dDb = 0.0; }
	_Map():pData(nullptr),pError(nullptr),dBias(0),dDb(0) {}
}Map,*PMap;

//网络层
typedef struct _Layer
{
	int nMapWidth;			//图像宽度
	int nMapHeight;			//图像高度
	int nMapCount;			//图像数量
	PMap pMap;				//图像指针

	int nKernelWidth;		//卷积核宽度
	int nKernelHeight;		//卷积核高度
	int nKernelCount;		//卷积核数量
	PKernel pKernel;		//卷积核指针

	double* pMapCommon;		//未知
	void Release() 
	{
		nMapWidth = nMapHeight = nMapCount = 0;
		nKernelWidth = nKernelHeight = nKernelCount = 0;
		pMapCommon = nullptr;
	}
	_Layer() :nMapWidth(0),nMapHeight(0),nMapCount(0),pMap(nullptr),
	nKernelWidth(0),nKernelHeight(0),nKernelCount(0),pKernel(nullptr),pMapCommon(nullptr){}
}Layer,*PLayer;

//Mnist卷积神经网络
typedef struct _MnistNet
{
	Layer stInputLayer_0;	//输入层	
	Layer stConvLayer_1;	//卷积层
	Layer stPoolLayer_2;	//池化层
	Layer stConvLayer_3;	//卷积层
	Layer stPoolLayer_4;	//池化层
	Layer stConvLayer_5;	//卷积层
	Layer stOutputLayer_6;	//输出层

}MnistNet,*PMnistNet;

//连接表
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

//Double的有效范围
inline bool IsValidDouble(double dValue) { return (dValue <= DBL_MAX && dValue >= -DBL_MAX); };

//初始化卷积核
bool InitializeKernel(double* pWeight,	//权重地址
	int nKernelSize,					//卷积核大小
	double dWeightBase);				//权重基准

//初始化网络层
bool InitializeLayer(Layer& stLayer,	//当前层
	int nPreviousLayerMapNumber,		//上一层图像数量
	int nOutputMapNumber,				//当前层输出图像数量
	int nKernelWidth,					//卷积核宽度
	int nKernelHeight,					//卷积核高度
	int nInputMapWidth,					//输入图像宽度
	int nInputMapHeight,				//输入图像高度
	bool bIsPooling = false);			//是否池化

//初始化网络
bool InitializeMnistNet(MnistNet& stMnistNet,	//结构指针
	int nWidth,									//输入图像宽度
	int nHeight,								//输入图像高度
	int nClassNumber);							//预测类别数量

//开始训练模型
bool TrainModel(MnistNet& stMnistNet,			//网络结构
	MnistData& stMnistTrain,					//训练集数据
	MnistData& stMnistTest,						//测试集数据
	double dLearningRate,						//学习率
	int nBatchSize,								//批量数量
	int nEpoch = 5);							//迭代次数

//重置权重
bool ResetWeight(MnistNet& stMnistNet);			//网络结构

//重置层
bool ResetLayer(Layer& stLayer);				//层结构

//更新权重
bool UpdateWeight(MnistNet& stMnistNet,			//网络结构
	double dLearnigrate,						//学习率
	int nBatchSize);							//批量数量

//更新层
bool UpdateLayer(Layer& stLayer,				//层结构
	double dLearningRate,						//学习率
	int nBatchSize);							//批量数量			

//梯度下降算法
double GradientDescent(double dWeight,			//当前权重
	double dWd,									//未知
	double dLearningRate,						//学习率
	double dLambda);							//未知

//前向传播
bool ForwardPropagation(MnistNet& stMnistNet);	//网络结构

//反向传播
bool BackwardPropagation(MnistNet& stMnistNet,	//网络结构
	double* pLabelData);						//标签数据

//卷积层的前向传播
bool ForwardToConvolution(Layer& stPreviouLayer,//前一层
	Layer& stCurrentLayer,						//当前层
	const bool* pConnectTable = nullptr);		//连接表指针

//池化层的前向传播,这里用的是最大池化
bool ForwardToPooling(Layer& stPreviouLayer,	//前一层
	Layer& stCurrentLayer);						//当前层

//全连接层的前向传播
bool ForwardToFullConnect(Layer& stPreviouLayer,//前一层
	Layer& stCurrentLayer);						//当前层

//有效卷积
bool ValidConvolution(double* pInputData,		//输入图像数据
	int nInputWidth,							//输入图像宽度
	int nInputHeight,							//输入图像高度
	double* pKernelData,						//卷积核数据
	int nKernelWidth,							//卷积核宽度
	int nKernelHeight,							//卷积核高度
	double* pOutputData,						//输出图像数据
	int nOutputWidth,							//输出图像宽度
	int nOuputHeight);							//输出图像高度

//激活函数Tanh
double ActivationTanh(double dValue);

//激活函数Tanh的导数
double DerivativeTanh(double dValue);

//激活函数Relu
double ActivationRelu(double dValue);

//激活函数Relu的导数
double DerivativeRelu(double dValue);

//激活函数Sigmoid
double ActivationSigmoid(double dValue);

//激活函数Sigmoid的导数
double DerivativeSigmoid(double dValue);

//全连接层的反向传播
bool BackwardToFullConnect(Layer& stCurrentLayer,	//当前层
	Layer& stPreviouLayer);							//上一层

//卷积层的反向传播
bool BackwardToConvolution(Layer& stCurrentLayer,	//当前层
	Layer& stPreviouLayer,							//上一层
	const bool* pConnectTable = nullptr);			//连接表

//池化层的反向传播
bool BackwardToPooling(Layer& stCurrentLayer,		//当前层
	Layer& stPreviouLayer);							//上一层

//模型预测
bool Predicts(MnistNet& stMnistNet,					//网络结构
	MnistData& stMnistData);						//Mnist数据

//获取输出值索引
int GetOutputIndex(Layer& stOutputLayer);			//输出层

//获取实际值索引
int GetActualIndex(double* pLabel,					//标签数据
	int nClassNumber);								//类别数量

//释放网络结构
bool ReleaseMnistNet(MnistNet& stMnistNet);			//网络结构

//释放层结构
bool ReleaseLayer(Layer& stLayer);					//层结构





