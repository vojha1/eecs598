#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
extern "C" {
	#include "header.h"
	#include "utilities.h"
}

//Aggregation function



//Combination function
__global__ void combination(float* in_feature, int fea_row, int fea_col,  float* weight, float* bias, int para_in, int para_out, float* out_feature, bool relu ){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    //For output feature matrix
    int x = tx + blockDim.x * bx; //Nodes
    int y = ty + blockDim.y * by; //Out Features
    float val=0;

    if(x<fea_col && y<para_out){
	    out_feature[y*fea_col+x] = bias[y];
	    for (int k = 0; k < para_in; ++k){
		    val += in_feature[k*fea_col + x] * weight[k*para_out + y];
		 //   printf("%d %lf\n",k,in_feature[k*fea_col + x]); //, weight[k*para_out + y]);
	    }
	   out_feature[y*fea_col+x] += val;
	   if(relu) out_feature[y*fea_col+x] = MAX(0.00000, out_feature[y*fea_col+x]);
    }
    //__syncthreads();
}

//Analyzer function


//Testing function
__global__ void check(){
   // printf("Thread %d %d from block %d %d \n",threadIdx.x,threadIdx.y,blockIdx.x, blockIdx.y);
}

int main(int argc, char const *argv[]) {
	if ((argc != 2) || ((strcmp(argv[1], "cora") != 0) && (strcmp(argv[1], "citeseer") != 0) && (strcmp(argv[1], "reddit") != 0))) {
		printf("ERROR: usage \"%s [cora|citeseer|reddit]\"\n", argv[0]);
		return -1;
	}
	GCN_t GCN_c = GCN_parser((char*)argv[1]);
	feature_t feature_c;	
	//CUDA Code section
	//Add commands for profiling -> Look into this -> Lec 17.pdf
	//Aggregation kernal

	//Combination kernal
	//This is only for testing 
	feature_c = aggregation(GCN_c.graph_c, GCN_c.feature_c);

	//Timing 
	struct timeval stop, start;
	gettimeofday(&start, NULL);
	
	//Get parameter sizes
	int l1_para_in = GCN_c.l1_parameter_c.in_feature_num;
	int l1_para_out =GCN_c.l1_parameter_c.out_feature_num;

	//Define and allocate  outputs for the combination kernal
	float *device_parameter_weight, *device_parameter_bias;
	cudaMalloc((void**)&device_parameter_weight, (l1_para_in*l1_para_out)*sizeof(float));
	cudaMalloc((void**)&device_parameter_bias, (l1_para_out)*sizeof(float));
	cudaMemcpy(device_parameter_weight,GCN_c.l1_parameter_c.weights[0] , (l1_para_in*l1_para_out)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_parameter_bias, GCN_c.l1_parameter_c.biasses, (l1_para_out)*sizeof(float), cudaMemcpyHostToDevice);

	//This doesn't assume that the output of the aggregation kernal is stored in CUDA	
	//Make sure that this allocation is not needed
	float *in_feature;
	cudaMalloc((void**)&in_feature, (feature_c.feature_num*feature_c.node_num)*sizeof(float));
	cudaMemcpy(in_feature,feature_c.features[0] , (feature_c.feature_num*feature_c.node_num)*sizeof(float), cudaMemcpyHostToDevice);
		
	//Define and allocate the output of the combination kernal
	float *out_feature;
	cudaMalloc((void**)&out_feature, (feature_c.node_num*l1_para_out)*sizeof(float)); 

	//Define the grid and block sizes and launch the kernal
	dim3 Dg( ceil(GCN_c.spec_c.nodes/32.0),ceil(l1_para_out/32.0)) ;
        dim3 Db(32,32,1);
	combination<<<Dg,Db>>>(in_feature,feature_c.feature_num,feature_c.node_num, device_parameter_weight,device_parameter_bias, l1_para_in, l1_para_out,out_feature,true);
	cudaDeviceSynchronize();
        
        gettimeofday(&stop, NULL);
	float secs1 = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
	
	//Checking the solution of the combination kernal
	//Remove later
	gettimeofday(&start, NULL);
	feature_t  feature_check = combination(feature_c, GCN_c.l1_parameter_c, true);
	gettimeofday(&stop, NULL);
	float secs2 = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
	
       	gettimeofday(&start, NULL);
	//Update feature_c and copy the feaatures values back into CPU
        feature_c.feature_num = l1_para_out;
	cudaMemcpy(feature_c.features[0],out_feature,(feature_c.node_num*l1_para_out)*sizeof(float), cudaMemcpyDeviceToHost); 
	gettimeofday(&stop, NULL);
	float secs3 = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);

	//Testing the combination kernal 
	// Remove later	
        //for(int i=0; i<100; i++){
        //      printf("%d %lf %lf \n",i,feature_check.features[4][i], feature_c.features[4][i]); //feature_c.features[13][i]);
        //}

	printf("CPU Time: %f sec\n",secs2);
	printf("GPU Time: %f sec\n",secs1+secs3);

	cudaFree(device_parameter_weight);
	cudaFree(device_parameter_bias);
	cudaFree(in_feature);
	cudaFree(out_feature);
	//Aggregation kernal


        //Combination kernal
	
	
	//Analyzer kernal
	feature_c = aggregation(GCN_c.graph_c, feature_c);
        
	//Timing
        gettimeofday(&start, NULL);

        //Get parameter sizes
        int l2_para_in = GCN_c.l2_parameter_c.in_feature_num;
        int l2_para_out =GCN_c.l2_parameter_c.out_feature_num;

        //Define and allocate  outputs for the combination kernal
        cudaMalloc((void**)&device_parameter_weight, (l2_para_in*l2_para_out)*sizeof(float));
        cudaMalloc((void**)&device_parameter_bias, (l2_para_out)*sizeof(float));
        cudaMemcpy(device_parameter_weight,GCN_c.l2_parameter_c.weights[0] , (l2_para_in*l2_para_out)*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_parameter_bias, GCN_c.l2_parameter_c.biasses, (l2_para_out)*sizeof(float), cudaMemcpyHostToDevice);

        //This doesn't assume that the output of the aggregation kernal is stored in CUDA
        //Make sure that this allocation is not needed
        cudaMalloc((void**)&in_feature, (feature_c.feature_num*feature_c.node_num)*sizeof(float));
        cudaMemcpy(in_feature,feature_c.features[0] , (feature_c.feature_num*feature_c.node_num)*sizeof(float), cudaMemcpyHostToDevice);

        //Define and allocate the output of the combination kernal
        cudaMalloc((void**)&out_feature, (feature_c.node_num*l2_para_out)*sizeof(float));

        //Define the grid and block sizes and launch the kernal
        dim3 Dg2( ceil(GCN_c.spec_c.nodes/32.0),ceil(l2_para_out/32.0)) ;
        combination<<<Dg2,Db>>>(in_feature,feature_c.feature_num,feature_c.node_num, device_parameter_weight,device_parameter_bias, l2_para_in, l2_para_out,out_feature,false);
        cudaDeviceSynchronize();

        gettimeofday(&stop, NULL);
        secs1 = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);

        //Checking the solution of the combination kernal
        //Remove later
        gettimeofday(&start, NULL);
        feature_check = combination(feature_c, GCN_c.l2_parameter_c, false);
        gettimeofday(&stop, NULL);
        secs2 = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);

        gettimeofday(&start, NULL);
        //Update feature_c and copy the feaatures values back into CPU
        feature_c.feature_num = l2_para_out;
        cudaMemcpy(feature_c.features[0],out_feature,(feature_c.node_num*l2_para_out)*sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&stop, NULL);
        secs3 = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);

        //Testing the combination kernal
        // Remove later
        //for(int i=0; i<100; i++){
        //      printf("%d %lf %lf \n",i,feature_check.features[4][i], feature_c.features[4][i]); //feature_c.features[13][i]);
        //}

        printf("CPU Time: %f sec\n",secs2);
        printf("GPU Time: %f sec\n",secs1+secs3);

        cudaFree(device_parameter_weight);
        cudaFree(device_parameter_bias);
        cudaFree(in_feature);
        cudaFree(out_feature);

	
	//feature_c = combination(feature_c, GCN_c.l2_parameter_c, false);
        analyzer(feature_c, GCN_c.label_c);
	return 0;
}
