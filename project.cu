#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern "C" {
	#include "header.h"
	#include "utilities.h"
}

//Aggregation function



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
	GCN_t GCN_c = GCN_parser((char*)argv[1]);
	feature_t feature_c; 
	//Add host and device variables here 


	
	//Serial code which will be used for comparision
	//feature_c = aggregation(GCN_c.graph_c, GCN_c.feature_c);
	//feature_c = combination(feature_c, GCN_c.l1_parameter_c, true);
	//feature_c = aggregation(GCN_c.graph_c, feature_c);
	//feature_c = combination(feature_c, GCN_c.l2_parameter_c, false);
	//analyzer(feature_c, GCN_c.label_c);
	//End of serial code

	
	//CUDA Code section
	
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
