#include "Net.h"
using namespace std;

int main(int argc, char* argv[])
{
	//训练集与测试集
	MnistData stTrainSet;
	MnistData stTestSet;

	//读取训练集数据
	ReadMnistData(stTrainSet, "E:\\Mnist\\New\\Train_Images");
	ReadMnistLable(stTrainSet, "E:\\Mnist\\New\\Train_Labels");

	//读取测试集数据
	ReadMnistData(stTestSet, "E:\\Mnist\\New\\Test_Images");
	ReadMnistLable(stTestSet, "E:\\Mnist\\New\\Test_Labels");

	//Mnist网络
	MnistNet stMnistNet;

	//初始化网络
	InitializeMnistNet(stMnistNet, stTrainSet.nWidth, stTrainSet.nHeight, stTrainSet.nClassNumber);

	//批量大小
	int nBatch_Size = 10;

	//学习率
	double dLearningRate = 0.01 * sqrt(nBatch_Size);

	//开始训练模型
	TrainModel(stMnistNet, stTrainSet, stTestSet, dLearningRate, nBatch_Size);
	
	//网络释放内存
	ReleaseMnistNet(stMnistNet);

	//释放Mnist数据
	ReleaseMnistData(stTrainSet);
	ReleaseMnistData(stTestSet);

	getchar();
	return 0;
}






