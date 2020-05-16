#define _SVID_SOURCE


#include "cuda_runtime.h"


#include "device_launch_parameters.h"

#include "omp.h"
#include "stdlib.h"
#include "stdio.h"
#include "time.h"



#pragma warning(disable : 4996)

#define BUFFER_SIZE 255

#define SIZE_CUDA_BLOCK 1024
#define SIZE_CUDA_THREAD_IN_BLOCK 1024


typedef struct Node Node;
typedef struct NodeList NodeList;

struct Node
{
	int value = -1;
	Node** paths;
	int sizePaths;
};

struct NodeList
{
	Node* node;
	NodeList* next;
};

struct Graph
{
	int size;
	Node* nodes;
};

struct Pointer
{
	Node* node;
	Pointer* next;
};


int stopwatchWork(int (*onWork)(int), int n) {

	double  startTime = omp_get_wtime();
	int result = (*onWork)(n);
	double endTime = omp_get_wtime();
	float timeWork = (endTime - startTime);
	printf("Time: %f \n", timeWork);

	return result;
}

// TODO need refactoring 
Graph createGraphFromFile(FILE* file) {
	if (file == NULL)
	{
		puts("File not find");
		exit(1);
	}
	int size;
	Graph graph = Graph();
	fread(&size, sizeof(size), 1, file);
	graph.size = size;
	int sizePaths;
	fread(&sizePaths, sizeof(int), 1, file);
	graph.nodes = new Node[graph.size];
	Node* nodes = graph.nodes;
	for (int i = 0; i < graph.size; i++)
	{
		Node node = Node();
		node.value = i;
		nodes[i] = node;
	}

	for (int row = 0; row < graph.size; row++)
	{
		Node* node = &nodes[row];
		int *hasPaths  = new int[graph.size];
		NodeList* firstNodeList = NULL;
		NodeList* lastNodeList = NULL;
		int sizeList = 0;
		fread(hasPaths, sizeof(int), graph.size, file);
		for (int column = 0; column < graph.size; column++)
		{
			if (hasPaths[column] != 0)
			{
				if (firstNodeList == NULL)
				{	
					firstNodeList = new NodeList();
					firstNodeList->node = &nodes[column];
					lastNodeList = firstNodeList;
				}
				else
				{
					NodeList *nextNode = new NodeList();
					nextNode->node = &nodes[column];
					lastNodeList->next = nextNode;
					lastNodeList = nextNode;
				}
				sizeList++;
			}
		}
		delete hasPaths;

		node->paths = new Node*[sizeList];
		node->sizePaths = sizeList;
		NodeList *currentNodeList = firstNodeList;
		for (int i = 0; i < node->sizePaths; i++)
		{
			node->paths[i] = currentNodeList->node;
			currentNodeList = currentNodeList->next;
		}
	}

	return graph;
}

int makeStep(Pointer* pointer, Pointer **lastPointer,Node* finishNode, int* pathCounter) {
	int counterNewPointer = 0;
	if (pointer != NULL && pointer->node->sizePaths > 0)
	{
		if (pointer->node == finishNode)
		{
			return counterNewPointer;
		}
		Node* tmpNode = pointer->node;
		Node* nextNode = tmpNode->paths[0];
		if (nextNode == finishNode)
		{
			(*pathCounter)++;
		}
		pointer->node = nextNode;


		for (int indexPath = 1; indexPath < tmpNode->sizePaths; indexPath++)
		{
			nextNode = tmpNode->paths[indexPath];
			if (nextNode == finishNode)
			{
				(*pathCounter)++;
			}
			(*lastPointer)->next = new Pointer();
			lastPointer = &(*lastPointer)->next;
			(*lastPointer)->node = nextNode;
			counterNewPointer++;
		}
	}
	return counterNewPointer;
}

int findAmountPathsGraph(Graph graph, int pointA, int pointB) {
	Node *nodeA = &graph.nodes[pointA];
	Node *nodeB = &graph.nodes[pointB];

	int amountPaths = 0;

	Pointer *movedPointers = new Pointer();
	movedPointers->node = nodeA;
	Pointer* lastPointer = movedPointers;
	int pointerSize = 1;
	for (int step = 0; step < graph.size; step++)
	{
		Pointer* currentPointer = movedPointers;
		int counterNewPointers = 0;
		for (int pointerIndex = 0; pointerIndex < pointerSize; pointerIndex++)
		{
			counterNewPointers +=  makeStep(currentPointer,&lastPointer, nodeB,&amountPaths);
			if (currentPointer == NULL)
			{
				break;
			}
			else
			{
				currentPointer = currentPointer->next;
			}
		}
		pointerSize += counterNewPointers;
	}


	return amountPaths;
}

__global__ void cudaStepAndNewPointers(int startindex, Node* tmpNode, Pointer** lastPointer, Node* finishNode, int* pathCounter, int* counterNewPointer) {
	int indexPath = (blockDim.x * blockIdx.x + threadIdx.x) + startindex;

	Node* nextNode = tmpNode->paths[indexPath];
	if (nextNode == finishNode)
	{
		atomicAdd(pathCounter, 1);
	}
	(*lastPointer)->next = new Pointer();
	lastPointer = &(*lastPointer)->next;
	(*lastPointer)->node = nextNode;
	atomicAdd(counterNewPointer, 1);
	__syncthreads();
}

__global__ void cudaStepForOne(Pointer* pointer, Node* finishNode, int* pathCounter) {
	Node* tmpNode = pointer->node;
	Node* nextNode = tmpNode->paths[0];
	if (nextNode == finishNode)
	{
		atomicAdd(pathCounter, 1);
	}
	pointer->node = nextNode;
	__syncthreads();

}

void cudaMakeStep(Pointer* pointer, Pointer** lastPointer, Node* finishNode, int* pathCounter, int* counterNewPointer) {
	if (pointer != NULL && pointer->node->sizePaths > 0)
	{
		if (pointer->node == finishNode)
		{
			return;
		}
		
		cudaStepForOne<<<1,1>>>(pointer, finishNode, pathCounter);

		Node* tmpNode = pointer->node;
		int sizePaths = tmpNode->sizePaths;
		int amountBlock = sizePaths / SIZE_CUDA_THREAD_IN_BLOCK;
		if ( sizePaths % SIZE_CUDA_THREAD_IN_BLOCK == 0)
		{
			cudaStepAndNewPointers<<<SIZE_CUDA_THREAD_IN_BLOCK, amountBlock>>>(1,tmpNode, lastPointer, finishNode, pathCounter, counterNewPointer);
		}
		else
		{
			if (amountBlock > 1)
			{
				cudaStepAndNewPointers <<<SIZE_CUDA_THREAD_IN_BLOCK, amountBlock >> > (1,tmpNode, lastPointer, finishNode, pathCounter, counterNewPointer);
				cudaStepAndNewPointers <<<sizePaths % SIZE_CUDA_THREAD_IN_BLOCK, 1 >> > (SIZE_CUDA_THREAD_IN_BLOCK * amountBlock + 1, tmpNode, lastPointer, finishNode, pathCounter, counterNewPointer);
			}
			else
			{
				cudaStepAndNewPointers << <sizePaths % SIZE_CUDA_THREAD_IN_BLOCK, 1 >> > (1, tmpNode, lastPointer, finishNode, pathCounter, counterNewPointer);
			}

		}
		
	}
}

int findGPU(Graph graph, int pointA, int pointB) {
	Graph* cudaGraph;
	cudaMalloc(&cudaGraph, sizeof(Graph));
	cudaMemcpy(cudaGraph, &graph, sizeof(graph), cudaMemcpyHostToDevice);




	cudaFree(cudaGraph);
}


int cudaFindAmountPathsGraph(Graph graph, int pointA, int pointB) {

	Node* nodeA = &graph.nodes[pointA];
	Node* nodeB = &graph.nodes[pointB];

	int amountPaths = 0;
	int* cudaResult;
	cudaMalloc(&cudaResult, sizeof(int));
	cudaMemcpy(cudaResult, &amountPaths, sizeof(int), cudaMemcpyHostToDevice);

	Pointer* movedPointers = new Pointer();
	movedPointers->node = nodeA;
	Pointer* lastPointer = movedPointers;
	int pointerSize = 1;
	int* cudaCounterNewPointers;
	cudaMalloc(&cudaCounterNewPointers, sizeof(int));
	for (int step = 0; step < graph.size; step++)
	{
		Pointer* currentPointer = movedPointers;
		int counterNewPointers = 0;
		

		for (int pointerIndex = 0; pointerIndex < pointerSize; pointerIndex++)
		{
			cudaMemcpy(cudaCounterNewPointers, 0, sizeof(int), cudaMemcpyHostToDevice);
			cudaMakeStep(currentPointer, &lastPointer, nodeB, cudaResult, cudaCounterNewPointers);
			if (currentPointer == NULL)
			{
				break;
			}
			currentPointer = currentPointer->next;
			int tmpCouter = 0;
			cudaMemcpy(&tmpCouter, cudaCounterNewPointers, sizeof(int), cudaMemcpyDeviceToHost);
			counterNewPointers += tmpCouter;
		}
		pointerSize += counterNewPointers;
	}

	cudaMemcpy(&amountPaths, cudaResult, sizeof(int), cudaMemcpyDeviceToHost);

	return amountPaths;
}

int main() {
	FILE* file;
	file = fopen("C:\\University\\Chydov\\PP\\laba2_graph\\laba2_graph\\x64\\Debug\\graph_6_5_7.bin", "rb");

	Graph graph= createGraphFromFile(file);
	
	int pointA;
	int pointB;

	printf("Input point A:");
	scanf("%d", &pointA);

	printf("Input point B:");
	scanf("%d", &pointB);

	int countPathsGPU = cudaFindAmountPathsGraph(graph, pointA, pointB);
	printf("GPU result: %d", countPathsGPU);

	int countPaths = findAmountPathsGraph(graph, pointA, pointB);
	printf("CPU result: %d", countPaths);

	

	return(0);
};



