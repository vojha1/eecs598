#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern "C" {
	#include "header.h"
	#include "utilities.h"
}

//Aggregation function
//total number of thread = number of features
//Each thread aggrgate particular feature for one node, and move to other feature
__global__ void aggregation_gpu(graph_t *graph_c, feature_t *in_feature_c, feature_t *out_feature_c){
	
	printf("in gpu check3\n");
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
    int feature_num = in_feature_c->feature_num;
    int node_num = in_feature_c->node_num;
    out_feature_c->feature_num = feature_num;
    out_feature_c->node_num = node_num;

    
    if(idx == 0)
		printf("in gpu check0\n");
	if(idx < feature_num) {
		for(int i = 0; i<node_num; ++i){
			for (int j = graph_c->indexes[i]; j < graph_c->indexes[i + 1]; ++j) {
				out_feature_c->features[idx * node_num + i] += in_feature_c->features[idx * node_num + graph_c->neighbours[j]];
			}
			out_feature_c->features[idx * node_num + i] /= ((float)graph_c->indexes[i + 1] - graph_c->indexes[i]);
		}
	}
	if(idx == 0)
		printf("in gpu check1\n");
}

//Combination function


//Analyzer function


//Testing function
__global__ void check(){
    printf("Thread %d from block %d \n",threadIdx.x, blockIdx.x);
}

int main(int argc, char const *argv[]) {
	if ((argc != 2) || ((strcmp(argv[1], "cora") != 0) && (strcmp(argv[1], "citeseer") != 0) && (strcmp(argv[1], "reddit") != 0))) {
		printf("ERROR: usage \"%s [cora|citeseer|reddit]\"\n", argv[0]);
		return -1;
	}
	printf("start paser\n");
	GCN_t GCN_c = GCN_parser((char*)argv[1]);
	feature_t feature_c; 
	//Add host and device variables here
	graph_t *device_graph_c; //device input graph (1st aggregation)
	feature_t *device_in_feature_c; //device input feature (1st aggregation)
	feature_t *device_out_feature_c; //device output feature (1st aggregation)
	feature_t *host_out_feature_c; //host output feature (1st aggregation)

	
	//Serial code which will be used for comparision
	//feature_c = aggregation(GCN_c.graph_c, GCN_c.feature_c);
	//feature_c = combination(feature_c, GCN_c.l1_parameter_c, true);
	//feature_c = aggregation(GCN_c.graph_c, feature_c);
	//feature_c = combination(feature_c, GCN_c.l2_parameter_c, false);
	//analyzer(feature_c, GCN_c.label_c);
	//End of serial code

	
	//CUDA Code section
	//1st aggregation start//
	printf("check0\n");
	int feature_num = GCN_c.feature_c.feature_num;
	int node_num = GCN_c.feature_c.node_num;
	int indexs_num = GCN_c.spec_c.nodes+1;
	int neighbours_num = GCN_c.spec_c.edges;
	cudaMalloc(&device_graph_c, sizeof(graph_t));
	cudaMalloc(&device_graph_c->indexes, indexs_num * sizeof(int));
	cudaMalloc(&device_graph_c->neighbours, neighbours_num * sizeof(int));
	cudaMemcpy(device_graph_c, &GCN_c.graph_c, sizeof(graph_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_graph_c->indexes, GCN_c.graph_c.indexes, indexs_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_graph_c->indexes, GCN_c.graph_c.indexes, indexs_num * sizeof(int), cudaMemcpyHostToDevice);
	printf("check, feature num = %d, node num = %d\n", feature_num, node_num);
	cudaMalloc(&device_in_feature_c, sizeof(feature_t));
	cudaMalloc(&device_in_feature_c->features,  feature_num * node_num * sizeof(float));
	printf("check1\n");
	cudaMemcpy(device_in_feature_c, &GCN_c.feature_c, sizeof(feature_t), cudaMemcpyHostToDevice);
	printf("check2\n");
	for(int i=0; i<10; ++i)
		printf("%f ", GCN_c.feature_c.features[i]);
	printf("\n");
	cudaMemcpy(device_in_feature_c->features, GCN_c.feature_c.features, feature_num * node_num * sizeof(float), cudaMemcpyHostToDevice);
	printf("check3\n");
	cudaMalloc(&device_out_feature_c, sizeof(feature_t));
	cudaMalloc(&device_out_feature_c->features,  feature_num * node_num * sizeof(float));

	host_out_feature_c = (feature_t*)malloc(sizeof(feature_t));
	host_out_feature_c->features = (float*)malloc(feature_num * node_num * sizeof(float));
	printf("check4\n");
	int block_size = 128;
	int gdx = feature_num/block_size;
	if(feature_num % block_size != 0) gdx++;
	dim3 gd(gdx, 1, 1);
	dim3 bd(block_size, 1, 1);
	aggregation_gpu<<<gd, bd>>>(device_graph_c, device_in_feature_c, device_out_feature_c);
	printf("check5\n");
	cudaError_t err = cudaGetLastError();

     if ( err != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       

        // Possibly: exit(-1) if program cannot continue....
     }
	cudaDeviceSynchronize();
	//copy back to host
	cudaMemcpy(host_out_feature_c, device_out_feature_c, sizeof(feature_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_out_feature_c->features, device_out_feature_c->features, feature_num * node_num * sizeof(float), cudaMemcpyDeviceToHost);
	// output the result to file to check the correctness
	FILE *result_file;
	result_file = fopen("/home/wftseng/EECS598_GPU_Final_Project/eecs598/result/first_aggregation_result_gpu", "w+");
	if(result_file == NULL)
		printf("error opening file in ./result/first_aggregation_result_gpu\n");
	for(int i=0; i<feature_num*node_num; ++i)
		fprintf(result_file, "%f\n", host_out_feature_c->features[i]);
	fclose(result_file);
	printf("finish_writing the result\n");
	//1st aggregation end//

	//Add commands for profiling -> Look into this -> Lec 17.pdf
	//CUDA_PROFILE
   
	// Memory allocation for inputs and intermediate outputs in GPU memory
	//cudaMalloc((void**)&, ()*sizeof());
	
	//Transfer information from one host to device 
	//cudaMemcpy()

	//GPU kernal calls
	dim3 Dg(1) ;
	dim3 Db(64);
	
	//Testing kernal to see if cuda is working correctly, comment out 
	check<<<Dg,Db>>>();	
	cudaDeviceSynchronize();
	
	//Aggregation kernal


	//Combination kernal


	//Aggregation kernal


        //Combination kernal
	
	
	//Analyzer kernal
	
	return 0;
}
