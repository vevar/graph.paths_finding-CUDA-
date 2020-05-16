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
	NodeList* paths;

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
		NodeList* firstNode = NULL;
		NodeList* lastNode = NULL;
		fread(hasPaths, sizeof(int), graph.size, file);
		for (int column = 0; column < graph.size; column++)
		{
			if (hasPaths[column] != 0)
			{
				if (firstNode == NULL)
				{	
					firstNode = new NodeList();
					firstNode->node = &nodes[column];
					node->paths = firstNode;
					lastNode = firstNode;
				}
				else
				{
					NodeList *nextNode = new NodeList();
					nextNode->node = &nodes[column];
					lastNode->next = nextNode;
					lastNode = nextNode;
				}
			}
		}
		delete hasPaths;
	}

	return graph;
}

int main() {
	FILE* file;
	file = fopen("C:\\Users\\webve\\Google Drive\\Universities\\Master Degree\\Chydov\\PP\\laba2_graph\\laba2_graph\\x64\\Debug\\graph_6_5_7.bin", "rb");

	Graph graph= createGraphFromFile(file);
	return(0);
};



