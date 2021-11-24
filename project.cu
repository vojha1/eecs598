#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern "C" {
	#include "header.h"
	#include "utilities.h"
}

#define AGGR_BLOCK_SIZE 4
#define NODE_NUM 2708
#define NEIGHBOURS 10556
//Aggregation function
//total number of thread = number of features
//Each thread aggrgate particular feature for one node, and move to other feature
// __global__ void aggregation_gpu(graph_t *graph_c, feature_t *in_feature_c, feature_t *out_feature_c){
__global__ void aggregation_gpu(int *graph_c_indexes, int *graph_c_neighbours, float *in_feature_data, float *out_feature_data, int feature_num_in, int node_num_in){	
	//printf("in gpu check3\n");
	
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
    int feature_num = feature_num_in;
	int node_num = node_num_in;
    
    // if(idx == 0)
	// 	printf("in gpu check0\n");
	if(idx < feature_num) {
		for(int i = 0; i<node_num; ++i){
			for (int j = graph_c_indexes[i]; j < graph_c_indexes[i + 1]; ++j) {
				out_feature_data[idx * node_num + i] += in_feature_data[idx * node_num + graph_c_neighbours[j]];
			}
			out_feature_data[idx * node_num + i] /= ((float)graph_c_indexes[i + 1] - graph_c_indexes[i]);
		}
	}
	// if(idx == 0)
	// 	printf("in gpu check1\n");
}

__global__ void aggregation_gpu_tile(int *graph_c_indexes, int *graph_c_neighbours, float *in_feature_data, float *out_feature_data, int feature_num_in, int node_num_in){	
	//BLOCK_SIZE for this version limit to 4 because of shared memroy size
	int tx = threadIdx.x;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
    int feature_num = feature_num_in;
    int node_num = node_num_in;

    __shared__ float in_feature_tile[AGGR_BLOCK_SIZE][NODE_NUM];

	if(idx < feature_num)
		for(int i=0; i<node_num; ++i){
			in_feature_tile[tx][i] = in_feature_data[idx * node_num + i];
		}
    __syncthreads ();

	if(idx < feature_num) {
		for(int i = 0; i<node_num; ++i){
			for (int j = graph_c_indexes[i]; j < graph_c_indexes[i + 1]; ++j) {
				out_feature_data[idx * node_num + i] += in_feature_tile[tx][graph_c_neighbours[j]];
			}
			out_feature_data[idx * node_num + i] /= ((float)graph_c_indexes[i + 1] - graph_c_indexes[i]);
		}
	}
}

__global__ void aggregation_gpu_tile_v2(int *graph_c_indexes, int *graph_c_neighbours, float *in_feature_data, float *out_feature_data, int feature_num_in, int node_num_in){	
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
    int feature_num = feature_num_in;
    int node_num = node_num_in;
	
    __shared__ int graph_c_neighbours_tile[NEIGHBOURS];

	
	for(int i=0; i<NEIGHBOURS; i+=blockDim.x){
		if(tx + i <  NEIGHBOURS)
			graph_c_neighbours_tile[tx+i] = graph_c_neighbours[tx+i];
	}
    __syncthreads ();

	if(idx < feature_num) {
		for(int i = 0; i<node_num; ++i){
			for (int j = graph_c_indexes[i]; j < graph_c_indexes[i + 1]; ++j) {
				out_feature_data[idx * node_num + i] += in_feature_data[idx * node_num + graph_c_neighbours_tile[j]];
			}
			out_feature_data[idx * node_num + i] /= ((float)graph_c_indexes[i + 1] - graph_c_indexes[i]);
		}
	}
}

__global__ void aggregation_gpu_tile_v3(int *graph_c_indexes, int *graph_c_neighbours, float *in_feature_data, float *out_feature_data, int feature_num_in, int node_num_in){	
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
    int feature_num = feature_num_in;
    int node_num = node_num_in;
	
    __shared__ int graph_c_neighbours_tile[NEIGHBOURS];

	for(int i=0; i<NEIGHBOURS; i+=blockDim.x){
		if(tx + i <  NEIGHBOURS)
			graph_c_neighbours_tile[tx+i] = graph_c_neighbours[tx+i];
	}
    __syncthreads ();
	
	for(int i=0; i<node_num; i+=blockDim.x){
		int id = tx + i;
		if(id < node_num){
			for (int j = graph_c_indexes[id]; j < graph_c_indexes[id + 1]; ++j) {
				out_feature_data[id + bx * node_num] += in_feature_data[bx * node_num + graph_c_neighbours_tile[j]];
			}
			out_feature_data[id + bx * node_num] /= ((float)graph_c_indexes[id + 1] - graph_c_indexes[id]);	
		}
	}

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
	int *device_graph_indexes; //device input graph indexs(1st aggregation)
	int *device_graph_neighbours; //device input graph indexs(1st aggregation)
	float *device_feature_data_in; //device input feature data(1st aggregation)
	float *device_feature_data_out; //device output feature data(1st aggregation)
	//float *host_feature_data_out;
	//feature_t *host_out_feature_c; //host output feature (1st aggregation)

	
	//Serial code which will be used for comparision
	//feature_c = aggregation(GCN_c.graph_c, GCN_c.feature_c);
	//feature_c = combination(feature_c, GCN_c.l1_parameter_c, true);
	//feature_c = aggregation(GCN_c.graph_c, feature_c);
	//feature_c = combination(feature_c, GCN_c.l2_parameter_c, false);
	//analyzer(feature_c, GCN_c.label_c);
	//End of serial code

	
	//CUDA Code section
	//1st aggregation start//
	// printf("check0\n");
	int feature_num = GCN_c.feature_c.feature_num;
	int node_num = GCN_c.feature_c.node_num;
	int indexs_num = GCN_c.spec_c.nodes+1;
	int neighbours_num = GCN_c.spec_c.edges;;

	feature_c.feature_num = feature_num;
	feature_c.node_num = node_num;
	feature_c.features = (float *)malloc(feature_num * node_num * sizeof(float));

	//host_feature_data_out = (float *)malloc(feature_num * node_num * sizeof(float));

	cudaMalloc(&device_graph_indexes, indexs_num * sizeof(int));
	cudaMalloc(&device_graph_neighbours, neighbours_num * sizeof(int));
	cudaMalloc(&device_feature_data_in, feature_num * node_num * sizeof(int));
	cudaMalloc(&device_feature_data_out, feature_num * node_num * sizeof(int));

	cudaMemcpy(device_graph_indexes, GCN_c.graph_c.indexes, indexs_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_graph_neighbours, GCN_c.graph_c.neighbours, neighbours_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_feature_data_in, GCN_c.feature_c.features, feature_num * node_num * sizeof(float), cudaMemcpyHostToDevice);

	//printf("check, feature num = %d, node num = %d\n", feature_num, node_num);
	// cudaMalloc(&device_in_feature_c, sizeof(feature_t));
	// cudaMalloc(&device_in_feature_c->features,  feature_num * node_num * sizeof(float));
	// printf("check1\n");
	// cudaMemcpy(device_in_feature_c, &GCN_c.feature_c, sizeof(feature_t), cudaMemcpyHostToDevice);
	// printf("check2\n");
	// for(int i=0; i<10; ++i)
	// 	printf("%f ", GCN_c.feature_c.features[i]);
	// printf("\n");
	// cudaMemcpy(device_in_feature_c->features, GCN_c.feature_c.features, feature_num * node_num * sizeof(float), cudaMemcpyHostToDevice);
	// printf("check3\n");
	// cudaMalloc(&device_out_feature_c, sizeof(feature_t));
	// cudaMalloc(&device_out_feature_c->features,  feature_num * node_num * sizeof(float));

	// host_out_feature_c = (feature_t*)malloc(sizeof(feature_t));
	// host_out_feature_c->features = (float*)malloc(feature_num * node_num * sizeof(float));
	
	int block_size = AGGR_BLOCK_SIZE; //set to 4 because of shared memory size limit
	int gdx = feature_num/block_size;
	// int gdx = feature_num; // for v3
	if(feature_num % block_size != 0) gdx++;
	dim3 gd(gdx, 1, 1);
	dim3 bd(block_size, 1, 1);

	aggregation_gpu<<<gd, bd>>>(device_graph_indexes, device_graph_neighbours, device_feature_data_in, device_feature_data_out, feature_num, node_num);
	// aggregation_gpu_tile<<<gd, bd>>>(device_graph_indexes, device_graph_neighbours, device_feature_data_in, device_feature_data_out, feature_num, node_num);
	// aggregation_gpu_tile_v2<<<gd, bd>>>(device_graph_indexes, device_graph_neighbours, device_feature_data_in, device_feature_data_out, feature_num, node_num);
	// aggregation_gpu_tile_v3<<<gd, bd>>>(device_graph_indexes, device_graph_neighbours, device_feature_data_in, device_feature_data_out, feature_num, node_num);
	cudaError_t err = cudaGetLastError();

     if ( err != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        // Possibly: exit(-1) if program cannot continue....
     }
	cudaDeviceSynchronize();
	cudaMemcpy(feature_c.features, device_feature_data_out, feature_num * node_num * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// for debug start//
	FILE *result_file;
	result_file = fopen("/home/wftseng/EECS598_GPU_Final_Project/eecs598/result/first_aggregation_result_gpu", "w+");
	
	if(result_file == NULL){
		printf("error opening file in ./result/first_aggregation_result_gpu\n");
		return -1;
	}
	for(int i=0; i<feature_num*node_num; ++i)
		fprintf(result_file, "%f\n", feature_c.features[i]);
	fclose(result_file);
	printf("finish_writing the result\n");
	// for debug end//

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
