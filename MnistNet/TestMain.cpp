#include "Net.h"
using namespace std;

int main(int argc, char* argv[])
{
	//ѵ��������Լ�
	MnistData stTrainSet;
	MnistData stTestSet;

	//��ȡѵ��������
	ReadMnistData(stTrainSet, "E:\\Mnist\\New\\Train_Images");
	ReadMnistLable(stTrainSet, "E:\\Mnist\\New\\Train_Labels");

	//��ȡ���Լ�����
	ReadMnistData(stTestSet, "E:\\Mnist\\New\\Test_Images");
	ReadMnistLable(stTestSet, "E:\\Mnist\\New\\Test_Labels");

	//Mnist����
	MnistNet stMnistNet;

	//��ʼ������
	InitializeMnistNet(stMnistNet, stTrainSet.nWidth, stTrainSet.nHeight, stTrainSet.nClassNumber);

	//������С
	int nBatch_Size = 10;

	//ѧϰ��
	double dLearningRate = 0.01 * sqrt(nBatch_Size);

	//��ʼѵ��ģ��
	TrainModel(stMnistNet, stTrainSet, stTestSet, dLearningRate, nBatch_Size);
	
	//�����ͷ��ڴ�
	ReleaseMnistNet(stMnistNet);

	//�ͷ�Mnist����
	ReleaseMnistData(stTrainSet);
	ReleaseMnistData(stTestSet);

	getchar();
	return 0;
}






