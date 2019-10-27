#include "Net.h"
using namespace std;

bool InitializeKernel(double* pWeight, 
	int nKernelSize,
	double dWeightBase)
{
	static int nScale = 5;
	for (int i = 0; i < nKernelSize; i++)
	{
		int nRandom = rand();
		double dTemp = static_cast<double>(nRandom % (nKernelSize * nScale));
		dTemp = (dTemp == 0) ? nScale : dTemp;
		dTemp = (dTemp > nKernelSize) ? nKernelSize / dTemp : nKernelSize / (dTemp + nKernelSize);
		pWeight[i] = dTemp * dWeightBase * 2.0;
		if (nRandom % 2) pWeight[i] = -pWeight[i];
		if (pWeight[i] > 1.0)pWeight[i] = sqrt(pWeight[i]);
	}

	return true;
}

bool InitializeLayer(Layer& stLayer,
	int nPreviousLayerMapNumber,
	int nOutputMapNumber,
	int nKernelWidth,
	int nKernelHeight,
	int nInputMapWidth,
	int nInputMapHeight,
	bool bIsPooling)
{
	int nInput = 4, nOutput = 1;

	//����Ҫ�ػ�
	if (!bIsPooling)
	{
		nInput = nPreviousLayerMapNumber * nKernelWidth * nKernelHeight;
		nOutput = nOutputMapNumber * nKernelWidth * nKernelHeight;
	}

	//Ȩ�ػ�׼
	double dWeightBase = (nInput + nOutput) ? sqrt(6.0 / static_cast<double>(nInput + nOutput)) : 0.5;

	//����ͼ����
	stLayer.nMapWidth = nInputMapWidth;
	stLayer.nMapHeight = nInputMapHeight;

	//����ͼ������
	stLayer.nMapCount = nOutputMapNumber;

	//�������˿��
	stLayer.nKernelWidth = nKernelWidth;
	stLayer.nKernelHeight = nKernelHeight;

	//���������
	stLayer.nKernelCount = nPreviousLayerMapNumber * nOutputMapNumber;
	if(stLayer.nKernelCount) stLayer.pKernel = new Kernel[stLayer.nKernelCount];

	int nKernelSize = nKernelWidth * nKernelHeight;

	for (int i = 0; i < nPreviousLayerMapNumber; i++)
	{
		for (int j = 0; j < nOutputMapNumber; j++)
		{
			//�����/Ȩ��
			if (nKernelSize)
			{
				stLayer.pKernel[i*nOutputMapNumber + j].pWeight = new double[nKernelSize];
				InitializeKernel(stLayer.pKernel[i*nOutputMapNumber + j].pWeight, nKernelSize, dWeightBase);

				//δ֪
				stLayer.pKernel[i*nOutputMapNumber + j].pDw = new double[nKernelSize];
				memset(stLayer.pKernel[i*nOutputMapNumber + j].pDw, 0, sizeof(double)*nKernelSize);
			}
		}
	}

	int nMapSize = nInputMapWidth * nInputMapHeight;
	stLayer.pMap = new Map[nOutputMapNumber];
	for (int i = 0; i < nOutputMapNumber; i++)
	{
		stLayer.pMap[i].dBias = 0.0;
		stLayer.pMap[i].dDb = 0.0;
		if (nMapSize)
		{
			stLayer.pMap[i].pData = new double[nMapSize];
			stLayer.pMap[i].pError = new double[nMapSize];
			memset(stLayer.pMap[i].pData, 0, sizeof(double)*nMapSize);
			memset(stLayer.pMap[i].pError, 0, sizeof(double)*nMapSize);
		}
	}

	if (nMapSize)
	{
		stLayer.pMapCommon = new double[nMapSize];
		memset(stLayer.pMapCommon, 0, sizeof(double)*nMapSize);
	}

	return true;
}

bool InitializeMnistNet(MnistNet& stMnistNet,int nWidth,int nHeight,int nClassNumber)
{
	//��ʼ��һ���������
	srand(static_cast<unsigned int>(chrono::system_clock::now().time_since_epoch().count()));

	//����˿�Ⱥ͸߶�
	int nKernelWidth = 0, nKernelHeight = 0;

	//��ʼ�������0
	InitializeLayer(stMnistNet.stInputLayer_0,
		0,
		1,
		nKernelWidth,
		nKernelHeight,
		nWidth,
		nHeight);

	//��ʼ�������1
	nKernelWidth = nKernelHeight = 5;
	InitializeLayer(stMnistNet.stConvLayer_1,
		1,
		6,
		nKernelWidth,
		nKernelHeight,
		stMnistNet.stInputLayer_0.nMapWidth - nKernelWidth + 1,
		stMnistNet.stInputLayer_0.nMapHeight - nKernelHeight + 1);

	//��ʼ���ػ���2
	nKernelWidth = nKernelHeight = 1;
	InitializeLayer(stMnistNet.stPoolLayer_2,
		1,
		6,
		nKernelWidth,
		nKernelHeight,
		stMnistNet.stConvLayer_1.nMapWidth / 2,
		stMnistNet.stConvLayer_1.nMapHeight / 2,
		true);

	//��ʼ�������3
	nKernelWidth = nKernelHeight = 5;
	InitializeLayer(stMnistNet.stConvLayer_3,
		6,
		16,
		nKernelWidth,
		nKernelHeight,
		stMnistNet.stPoolLayer_2.nMapWidth - nKernelWidth + 1,
		stMnistNet.stPoolLayer_2.nMapHeight - nKernelHeight + 1);

	//��ʼ���ػ���4
	nKernelWidth = nKernelHeight = 1;
	InitializeLayer(stMnistNet.stPoolLayer_4,
		6,
		16,
		nKernelWidth,
		nKernelHeight,
		stMnistNet.stConvLayer_3.nMapWidth / 2,
		stMnistNet.stConvLayer_3.nMapHeight / 2,
		true);

	//��ʼ�������5
	nKernelWidth = nKernelHeight = 5;
	InitializeLayer(stMnistNet.stConvLayer_5,
		16,
		120,
		nKernelWidth,
		nKernelHeight,
		stMnistNet.stPoolLayer_4.nMapWidth - nKernelWidth + 1,
		stMnistNet.stPoolLayer_4.nMapHeight - nKernelHeight + 1);

	//��ʼ�������6
	nKernelWidth = nKernelHeight = 1;
	InitializeLayer(stMnistNet.stOutputLayer_6,
		120,
		nClassNumber,
		nKernelWidth,
		nKernelHeight,
		1,
		1);

	return true;
}

bool TrainModel(MnistNet& stMnistNet,
	MnistData& stMnistTrain,
	MnistData& stMnistTest,
	double dLearningRate,
	int nBatchSize,
	int nEpoch)
{
	Time cTime;
	
	int* pRamdomSort = new int[stMnistTrain.nNumber];

	//�����õ���С�����ݶ��½��㷨
	int nBatchNumber = stMnistTrain.nNumber / nBatchSize;

	for (int i = 0; i < nEpoch; i++)
	{
		//��������
		for (int k = 0; k < stMnistTrain.nNumber; k++)
			pRamdomSort[k] = k;

		//�������
		for (int k = 0; k < stMnistTrain.nNumber; k++)
		{
			int nSortIndex = rand() % (stMnistTrain.nNumber - k) + k;
			int nValue = pRamdomSort[nSortIndex];
			pRamdomSort[nSortIndex] = pRamdomSort[k];
			pRamdomSort[k] = nValue;
		}

		int nFinishingRate = 0;
		cout << endl;
		cout << "-----------------------------------------------------------------------------------" << endl;
		cout << "Epoch:[" << i << "] - ";
		cout << "TrainSet Number:[" << stMnistTrain.nNumber << "] - ";
		cout << "Mini Batch Size:[" << nBatchSize << "] - ";
		cout << "Learning Rate:[" << dLearningRate << "] - " << endl;
		cout << "��FinishingRate��:";

		cTime.ReSetTime();

		for (int j = 0; j < nBatchNumber; j++)
		{
			//����Ȩ��
			ResetWeight(stMnistNet);

			for (int k = 0; k < nBatchSize; k++)
			{
				int nIndex = j * nBatchSize + k;
				memcpy_s(stMnistNet.stInputLayer_0.pMap[0].pData,
					sizeof(double) * stMnistNet.stInputLayer_0.nMapWidth * stMnistNet.stInputLayer_0.nMapHeight,
					stMnistTrain.pData[pRamdomSort[nIndex]],
					sizeof(double) * stMnistTrain.nWidth * stMnistTrain.nHeight);

				//ǰ�򴫲�
				ForwardPropagation(stMnistNet);

				//���򴫲�
				BackwardPropagation(stMnistNet, stMnistTrain.pLable[pRamdomSort[nIndex]]);

				//��ʾ�ٷֱ�
				if (nIndex && (nIndex % (stMnistTrain.nNumber / 10)) == 0)
				{
					nFinishingRate += 10;
					if (nFinishingRate < 90)
						cout << nFinishingRate << "% -> ";
					else
						cout << nFinishingRate << "%...." << endl;
				}
			}

			//����Ȩ��
			UpdateWeight(stMnistNet, dLearningRate, nBatchSize);
		}

		cout << "Total Training Time:[" << cTime.GetTimeCount() << "]Minutes..." << endl;

		//Ԥ��
		Predicts(stMnistNet, stMnistTest);

		cout << "Epoch:[" << i << "] - End Iteration Training...." << endl;
		cout << "-----------------------------------------------------------------------------------" << endl;

		//����ѧϰ��
		dLearningRate *= 0.85;
	}

	delete[] pRamdomSort;

	return true;
}

bool ResetWeight(MnistNet& stMnistNet)
{
	//ResetLayer(stMnistNet.stInputLayer_0);
	ResetLayer(stMnistNet.stConvLayer_1);
	ResetLayer(stMnistNet.stPoolLayer_2);
	ResetLayer(stMnistNet.stConvLayer_3);
	ResetLayer(stMnistNet.stPoolLayer_4);
	ResetLayer(stMnistNet.stConvLayer_5);
	ResetLayer(stMnistNet.stOutputLayer_6);

	return true;
}

bool ResetLayer(Layer& stLayer)
{
	//�����
	for (int i = 0; i < stLayer.nKernelCount; i++)
	{
		memset(stLayer.pKernel[i].pDw, 0, sizeof(double)*stLayer.nKernelWidth*stLayer.nKernelHeight);
	}

	//ͼ������
	for (int i = 0; i < stLayer.nMapCount; i++)
	{
		stLayer.pMap[i].dDb = 0.0;
	}

	return true;
}

bool UpdateWeight(MnistNet& stMnistNet, 
	double dLearnigrate,
	int nBatchSize)
{
	//UpdateLayer(stMnistNet.stInputLayer_0, dLearnigrate, nBatchSize);
	UpdateLayer(stMnistNet.stConvLayer_1, dLearnigrate, nBatchSize);
	UpdateLayer(stMnistNet.stPoolLayer_2, dLearnigrate, nBatchSize);
	UpdateLayer(stMnistNet.stConvLayer_3, dLearnigrate, nBatchSize);
	UpdateLayer(stMnistNet.stPoolLayer_4, dLearnigrate, nBatchSize);
	UpdateLayer(stMnistNet.stConvLayer_5, dLearnigrate, nBatchSize);
	UpdateLayer(stMnistNet.stOutputLayer_6, dLearnigrate, nBatchSize);

	return true;
}

bool UpdateLayer(Layer& stLayer, 
	double dLearningRate,
	int nBatchSize)
{
	static double dLambda = 0.005;

	//Ȩ��
	for (int i = 0; i < stLayer.nKernelCount; i++)
	{
		for (int j = 0; j < stLayer.nKernelWidth * stLayer.nKernelHeight; j++)
		{
			double dTemp = GradientDescent(stLayer.pKernel[i].pWeight[j], stLayer.pKernel[i].pDw[j] / nBatchSize, dLearningRate, dLambda);
			stLayer.pKernel[i].pWeight[j] = dTemp;
		}
	}

	//ƫ��
	for (int i = 0; i < stLayer.nMapCount; i++)
	{
		double dTemp = GradientDescent(stLayer.pMap[i].dBias, stLayer.pMap[i].dDb / nBatchSize, dLearningRate, dLambda);
		stLayer.pMap[i].dBias = dTemp;
	}

	return true;
}

double GradientDescent(double dWeight,
	double dWd,
	double dLearningRate,
	double dLambda)
{
	return dWeight - dLearningRate * (dWd + dLambda * dWeight);
}

bool ForwardPropagation(MnistNet& stMnistNet)
{
	//�����0 -> �����1
	ForwardToConvolution(stMnistNet.stInputLayer_0, stMnistNet.stConvLayer_1);

	//�����1 -> �ػ���2
	ForwardToPooling(stMnistNet.stConvLayer_1, stMnistNet.stPoolLayer_2);

	//�ػ���2 -> �����3
	ForwardToConvolution(stMnistNet.stPoolLayer_2, stMnistNet.stConvLayer_3, NetConnectTable);

	//�����3 -> �ػ���4
	ForwardToPooling(stMnistNet.stConvLayer_3, stMnistNet.stPoolLayer_4);

	//�ػ���4 -> �����5
	ForwardToConvolution(stMnistNet.stPoolLayer_4, stMnistNet.stConvLayer_5);

	//�����5 -> �����
	ForwardToFullConnect(stMnistNet.stConvLayer_5, stMnistNet.stOutputLayer_6);

	return true;
}

bool BackwardPropagation(MnistNet& stMnistNet,
	double* pLabelData)
{
	
	for (int i = 0; i < stMnistNet.stOutputLayer_6.nMapCount; i++)
	{
		//�������ֵ��ʵ��ֵ�����
		double dValue = (stMnistNet.stOutputLayer_6.pMap[i].pData[0] - pLabelData[i]);

		//����Tanh�ĵ����õ����
		dValue *= DerivativeTanh(stMnistNet.stOutputLayer_6.pMap[i].pData[0]);
		stMnistNet.stOutputLayer_6.pMap[i].pError[0] = dValue;
	}

	//�����6 -> �����5
	BackwardToFullConnect(stMnistNet.stOutputLayer_6, stMnistNet.stConvLayer_5);

	//�����5 -> �ػ���4
	BackwardToConvolution(stMnistNet.stConvLayer_5,stMnistNet.stPoolLayer_4);

	//�ػ���4 -> �����3
	BackwardToPooling(stMnistNet.stPoolLayer_4,stMnistNet.stConvLayer_3);

	//�����3 -> �ػ���2
	BackwardToConvolution(stMnistNet.stConvLayer_3, stMnistNet.stPoolLayer_2, NetConnectTable);

	//�ػ���2 -> �����1
	BackwardToPooling(stMnistNet.stPoolLayer_2, stMnistNet.stConvLayer_1);

	//�ػ���1 -> �����0
	BackwardToConvolution(stMnistNet.stConvLayer_1, stMnistNet.stInputLayer_0);

	return true;
}

bool ForwardToConvolution(Layer& stPreviouLayer,
	Layer& stCurrentLayer,
	const bool* pConnectTable)
{
	int nMapSize = stCurrentLayer.nMapWidth * stCurrentLayer.nMapHeight;
	int nIndex = 0;

	for (int i = 0; i < stCurrentLayer.nMapCount; i++)
	{
		//���
		memset(stCurrentLayer.pMapCommon, 0, sizeof(double)*nMapSize);

		for (int j = 0; j < stPreviouLayer.nMapCount; j++)
		{
			nIndex = j * stCurrentLayer.nMapCount + i;
			if (pConnectTable != nullptr && !pConnectTable[nIndex]) 
				continue;

			//��Ч���
			ValidConvolution(stPreviouLayer.pMap[j].pData,
				stPreviouLayer.nMapWidth,
				stPreviouLayer.nMapHeight,
				stCurrentLayer.pKernel[nIndex].pWeight,
				stCurrentLayer.nKernelWidth,
				stCurrentLayer.nKernelHeight,
				stCurrentLayer.pMapCommon,
				stCurrentLayer.nMapWidth,
				stCurrentLayer.nMapHeight);
		}

		//������ӳ��
		for (int k = 0; k < nMapSize; k++)
			stCurrentLayer.pMap[i].pData[k] = ActivationTanh(stCurrentLayer.pMapCommon[k] + stCurrentLayer.pMap[i].dBias);

	}

	return true;
}

bool ForwardToPooling(Layer& stPreviouLayer, 
	Layer& stCurrentLayer)
{
	for (int k = 0; k < stCurrentLayer.nMapCount; k++)
	{
		for (int i = 0; i < stCurrentLayer.nMapHeight; i++)
		{
			for (int j = 0; j < stCurrentLayer.nMapWidth; j++)
			{
				double dMax = stPreviouLayer.pMap[k].pData[2 * i*stPreviouLayer.nMapWidth + 2*j];

				for (int n = i * 2; n < 2 * (i + 1); n++)
				{
					for (int m = j * 2; m < 2 * (j + 1); m++)
					{
						double dTemp = stPreviouLayer.pMap[k].pData[n*stPreviouLayer.nMapWidth + m];
						if (dTemp > dMax) dMax = dTemp;
					}
				}

				stCurrentLayer.pMap[k].pData[i*stCurrentLayer.nMapWidth + j] = ActivationTanh(dMax);
			}
		}
	}

	return true;
}

bool ForwardToFullConnect(Layer& stPreviouLayer,
	Layer& stCurrentLayer)
{
	for (int i = 0; i < stCurrentLayer.nMapCount; i++)
	{
		double dSum = 0.0;
		for (int j = 0; j < stPreviouLayer.nMapCount; j++)
		{
			dSum += stPreviouLayer.pMap[j].pData[0] * stCurrentLayer.pKernel[j*stCurrentLayer.nMapCount + i].pWeight[0];
		}

		dSum += stCurrentLayer.pMap[i].dBias;
		stCurrentLayer.pMap[i].pData[0] = ActivationTanh(dSum);
	}

	return true;
}

bool ValidConvolution(double* pInputData, 
	int nInputWidth, 
	int nInputHeight, 
	double* pKernelData, 
	int nKernelWidth,
	int nKernelHeight, 
	double* pOutputData,
	int nOutputWidth, 
	int nOuputHeight)
{
	//�������ͼ���ڽ��в���
	double dSum;

	//�����i��j����ͼ���
	for (int i = 0; i < nOuputHeight; i++)
	{
		for (int j = 0; j < nOutputWidth; j++)
		{
			dSum = 0.0;
			//�����n��m�������˵�
			for (int n = 0; n < nKernelHeight; n++)
			{
				for (int m = 0; m < nKernelWidth; m++)
				{
					dSum += pInputData[(i + n)*nInputWidth + j + m] * pKernelData[n*nKernelWidth + m];
				}
			}
			//����ΪʲôҪ�ü��أ�����ΪBP��ʱ����Ҫ
			pOutputData[i*nOutputWidth + j] += dSum;
		}
	}

	return true;
}

double ActivationTanh(double dValue)
{
	double _dValue1 = exp(dValue);
	double _dValue2 = exp(-dValue);

	return (_dValue1 - _dValue2) / (_dValue1 + _dValue2);
}

double DerivativeTanh(double dValue)
{
	return 1.0 - dValue * dValue;
}

double ActivationRelu(double dValue)
{
	return (dValue > 0.0) ? dValue : 0.0;
}

double DerivativeRelu(double dValue)
{
	return (dValue > 0.0) ? 1.0 : 0.0;
}

double ActivationSigmoid(double dValue)
{
	return 1.0 / (1.0 + exp(-dValue));
}

double DerivativeSigmoid(double dValue)
{
	return dValue * (1.0 - dValue);
}

bool BackwardToFullConnect(Layer& stCurrentLayer, 
	Layer& stPreviouLayer)
{
	//�����
	for (int i = 0; i < stPreviouLayer.nMapCount; i++)
	{
		stPreviouLayer.pMap[i].pError[0] = 0.0;
		for (int j = 0; j < stCurrentLayer.nMapCount; j++)
		{
			double dValue = stCurrentLayer.pMap[j].pError[0] * stCurrentLayer.pKernel[i*stCurrentLayer.nMapCount + j].pWeight[0];
			stPreviouLayer.pMap[i].pError[0] += dValue;
		}
		stPreviouLayer.pMap[i].pError[0] *= DerivativeTanh(stPreviouLayer.pMap[i].pData[0]);
	}

	//DW
	for (int i = 0; i < stPreviouLayer.nMapCount; i++)
	{
		for (int j = 0; j < stCurrentLayer.nMapCount; j++)
		{
			stCurrentLayer.pKernel[i*stCurrentLayer.nMapCount + j].pDw[0] += stCurrentLayer.pMap[j].pError[0] * stPreviouLayer.pMap[i].pData[0];
		}
	}

	//�����
	for (int i = 0; i < stCurrentLayer.nMapCount; i++)
	{
		stCurrentLayer.pMap[i].dDb += stCurrentLayer.pMap[i].pError[0];
	}

	return true;
}

bool BackwardToConvolution(Layer& stCurrentLayer, 
	Layer& stPreviouLayer, 
	const bool* pConnectTable)
{

	for (int i = 0; i < stPreviouLayer.nMapCount; i++)
	{
		//�������
		memset(stPreviouLayer.pMapCommon,0,sizeof(double)*stPreviouLayer.nMapWidth * stPreviouLayer.nMapHeight);

		for (int j = 0; j < stCurrentLayer.nMapCount; j++)
		{
			int nIndex = i * stCurrentLayer.nMapCount + j;
			if (pConnectTable != nullptr && !pConnectTable[nIndex])
				continue;

			for (int n = 0; n < stCurrentLayer.nMapHeight; n++)
			{
				for (int m = 0; m < stCurrentLayer.nMapWidth; m++)
				{
					double dError = stCurrentLayer.pMap[j].pError[n*stCurrentLayer.nMapWidth + m];

					for (int y = 0; y < stCurrentLayer.nKernelHeight; y++)
					{
						for (int x = 0; x < stCurrentLayer.nKernelWidth; x++)
						{
							double dValue = dError * stCurrentLayer.pKernel[nIndex].pWeight[y*stCurrentLayer.nKernelWidth + x];
							stPreviouLayer.pMapCommon[(n + y)*stPreviouLayer.nMapWidth + m + x] += dValue;
						}
					}
				}
			}
		}

		for (int k = 0; k < stPreviouLayer.nMapHeight*stPreviouLayer.nMapWidth; k++)
			stPreviouLayer.pMap[i].pError[k] = stPreviouLayer.pMapCommon[k] * DerivativeTanh(stPreviouLayer.pMap[i].pData[k]);

	}

	//DW
	for (int i = 0; i < stPreviouLayer.nMapCount; i++)
	{
		for (int j = 0; j < stCurrentLayer.nMapCount; j++)
		{
			int nIndex = i * stCurrentLayer.nMapCount + j;
			if (pConnectTable !=nullptr && !pConnectTable[nIndex])
				continue;

			ValidConvolution(stPreviouLayer.pMap[i].pData,
				stPreviouLayer.nMapWidth,
				stPreviouLayer.nMapHeight,
				stCurrentLayer.pMap[j].pError,
				stCurrentLayer.nMapWidth,
				stCurrentLayer.nMapHeight,
				stCurrentLayer.pKernel[nIndex].pDw,
				stCurrentLayer.nKernelWidth,
				stCurrentLayer.nKernelHeight);
		}
	}

	//�����
	for (int i = 0;i < stCurrentLayer.nMapCount; i++)
	{
		double dSum = 0.0;
		for (int k = 0; k < stCurrentLayer.nMapWidth * stCurrentLayer.nMapHeight; k++)
		{
			dSum += stCurrentLayer.pMap[i].pError[k];
		}
		stCurrentLayer.pMap[i].dDb += dSum;
	}

	return true;
}

bool BackwardToPooling(Layer& stCurrentLayer, 
	Layer& stPreviouLayer)
{
	for (int k = 0; k < stCurrentLayer.nMapCount; k++)
	{
		for (int i = 0; i < stCurrentLayer.nMapHeight; i++)
		{
			for (int j = 0; j < stCurrentLayer.nMapWidth; j++)
			{
				int nHeight = 2 * i, nWidth = 2 * j;
				double dMax = stPreviouLayer.pMap[k].pData[nHeight*stPreviouLayer.nMapWidth + nWidth];
				for (int n = i * 2; n < 2 * (i + 1); n++)
				{
					for (int m = j * 2; m < 2 * (j + 1); m++)
					{
						if (stPreviouLayer.pMap[k].pData[n*stPreviouLayer.nMapWidth + m] > dMax)
						{
							nHeight = n;
							nWidth = m;
							dMax = stPreviouLayer.pMap[k].pData[n*stPreviouLayer.nMapWidth + m];
						}
						else
							stPreviouLayer.pMap[k].pError[n*stPreviouLayer.nMapWidth + m] = 0.0;
					}
				}
				double dValue = stCurrentLayer.pMap[k].pError[i*stCurrentLayer.nMapWidth + j] * DerivativeTanh(dMax);
				stPreviouLayer.pMap[k].pError[nHeight * stPreviouLayer.nMapWidth + nWidth] = dValue;
			}
		}
	}

	return true;
}

bool Predicts(MnistNet& stMnistNet,MnistData& stMnistData)
{
	//�������
	int nMatrixSize = stMnistData.nClassNumber*stMnistData.nClassNumber;
	int *pResultMatrix = new int[nMatrixSize];
	memset(pResultMatrix, 0, sizeof(int)*nMatrixSize);

	//�ɹ�Ԥ�������
	int nSucceesNumber = 0;

	Time cTime;

	for (int i = 0; i < stMnistData.nNumber; i++)
	{
		//��ͼ�����ݸ��Ƶ������
		memcpy_s(stMnistNet.stInputLayer_0.pMap[0].pData,
			sizeof(double)*stMnistData.nWidth * stMnistData.nHeight,
			stMnistData.pData[i],
			sizeof(double)*stMnistData.nWidth * stMnistData.nHeight);

		//ǰ�򴫲�
		ForwardPropagation(stMnistNet);

		//Ԥ��ֵ����
		int nPredictsIndex = GetOutputIndex(stMnistNet.stOutputLayer_6);

		//ʵ��ֵ����
		int nActualIndex = GetActualIndex(stMnistData.pLable[i], stMnistData.nClassNumber);

		//���ֵ������Ԥ��ֵ�����Ա�
		if (nPredictsIndex == nActualIndex)nSucceesNumber++;

		//�����¼
		pResultMatrix[nPredictsIndex * stMnistData.nClassNumber + nActualIndex]++;
	}

	cout << "Total Prediction Time:[" << cTime.GetTimeCount() << "]Minutes..." << endl;
	cout << "Class Type Number:" << stMnistData.nClassNumber << "\t";
	cout << "TestSet Number:" << stMnistData.nNumber << "\t";
	cout << "Success Number:" << nSucceesNumber << "\t";
	cout << "Accuracy:" << static_cast<double>(nSucceesNumber) / static_cast<double>(stMnistData.nNumber) * 100.0 << "%" << endl;

	cout << "Detail Matrix:" << endl;
	for (int i = 0; i < stMnistData.nClassNumber; i++)
	{
		cout << "[" << i << "]:" << "\t";
		for (int j = 0; j < stMnistData.nClassNumber; j++)
		{
			cout << pResultMatrix[i * stMnistData.nClassNumber + j] << "\t";
		}
		cout << endl;
	}
	cout << endl;

	delete[] pResultMatrix;
	return true;
}

int GetOutputIndex(Layer& stOutputLayer)
{
	//�������Ԥ��ֵ
	double dMaxValue = stOutputLayer.pMap[0].pData[0];
	//�������Ԥ��ֵ����
	int nMaxIndex = 0;

	//�������Ԥ��ֵ����
	for (int i = 1; i < stOutputLayer.nMapCount; i++)
	{
		if (stOutputLayer.pMap[i].pData[0] > dMaxValue)
		{
			dMaxValue = stOutputLayer.pMap[i].pData[0];
			nMaxIndex = i;
		}
	}

	return nMaxIndex;
}

int GetActualIndex(double* pLabel, 
	int nClassNumber)
{
	//����ʵ�����ֵ����
	int nMaxIndex = 0;
	//��ɫʵ�����ֵ����
	double dMaxValue = pLabel[0];

	//�������ֵ
	for (int i = 1; i < nClassNumber; i++)
	{
		if (pLabel[i] > dMaxValue)
		{
			dMaxValue = pLabel[i];
			nMaxIndex = i;
		}
	}

	return nMaxIndex;
}

bool ReleaseMnistNet(MnistNet& stMnistNet)
{
	ReleaseLayer(stMnistNet.stInputLayer_0);
	ReleaseLayer(stMnistNet.stConvLayer_1);
	ReleaseLayer(stMnistNet.stPoolLayer_2);
	ReleaseLayer(stMnistNet.stConvLayer_3);
	ReleaseLayer(stMnistNet.stPoolLayer_4);
	ReleaseLayer(stMnistNet.stConvLayer_5);
	ReleaseLayer(stMnistNet.stOutputLayer_6);

	return true;
}

bool ReleaseLayer(Layer& stLayer)
{
	//�ͷž����
	if (stLayer.pKernel)
	{
		for (int i = 0; i < stLayer.nKernelCount; i++)
		{
			if(stLayer.pKernel[i].pWeight)
				delete[] stLayer.pKernel[i].pWeight;
			if(stLayer.pKernel[i].pDw)
				delete[] stLayer.pKernel[i].pDw;
		}
		delete[] stLayer.pKernel;
	}

	//�ͷ�ͼ��
	if (stLayer.pMap)
	{
		for (int i = 0; i < stLayer.nMapCount; i++)
		{
			if (stLayer.pMap[i].pData)
				delete[] stLayer.pMap[i].pData;
			if (stLayer.pMap[i].pError)
				delete[] stLayer.pMap[i].pError;
		}
		delete[] stLayer.pMap;
	}

	if (stLayer.pMapCommon)
		delete[] stLayer.pMapCommon;

	//ָ��ȫ������Ϊnullptr
	stLayer.Release();

	return true;
}
