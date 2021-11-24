#include "header.h"

feature_t aggregation (graph_t graph_c, feature_t in_feature_c) {
	int i, k, j;
	feature_t out_feature_c;

	printf("AGGREGATION: A[%d][%d] * X[%d][%d] = X'[%d][%d]\n", in_feature_c.node_num, in_feature_c.node_num, in_feature_c.node_num, in_feature_c.feature_num, in_feature_c.node_num, in_feature_c.feature_num);

	out_feature_c.feature_num = in_feature_c.feature_num;
	out_feature_c.node_num = in_feature_c.node_num;
	out_feature_c.features = (float**) malloc (in_feature_c.feature_num * sizeof(float*));
	for (i = 0; i < in_feature_c.feature_num; ++i) {
		out_feature_c.features[i] = (float*) malloc (in_feature_c.node_num * sizeof(float));
		for (j = 0; j < in_feature_c.node_num; ++j) {
			out_feature_c.features[i][j] = 0;
		}
	}

	for (i = 0; i < in_feature_c.node_num; ++i) {
		printf("\r%.2f%% Completed!", (float)i * 100.00 / (float)in_feature_c.node_num);
	    fflush(stdout);
		for (k = 0; k < in_feature_c.feature_num; ++k) {
			for (j = graph_c.indexes[i]; j < graph_c.indexes[i + 1]; ++j) {
				out_feature_c.features[k][i] += in_feature_c.features[k][graph_c.neighbours[j]];
			}
			out_feature_c.features[k][i] /= ((float)graph_c.indexes[i + 1] - graph_c.indexes[i]);
		}
	}
	printf("\r\t\t\t\t\t\r");

	return out_feature_c;
}

feature_t combination (feature_t in_feature_c, parameter_t parameter_c, bool relu) {
	int i, j, k;
	feature_t out_feature_c;

	if (in_feature_c.feature_num != parameter_c.in_feature_num) {
    	printf("ERROR: Incompatible number of features in feature (%d) and parameter (%d) objects!\n", in_feature_c.feature_num, parameter_c.in_feature_num);
    	exit(-1);
	}

	printf("COMBINATION: X'[%d][%d] * W[%d][%d] = X[%d][%d]\n", in_feature_c.node_num, in_feature_c.feature_num, parameter_c.in_feature_num, parameter_c.out_feature_num, in_feature_c.node_num, parameter_c.out_feature_num);

	out_feature_c.node_num = in_feature_c.node_num;
	out_feature_c.feature_num = parameter_c.out_feature_num;
	out_feature_c.features = (float**) malloc (parameter_c.out_feature_num * sizeof(float*));
	float *t0 = (float*) malloc (parameter_c.out_feature_num *in_feature_c.node_num * sizeof(float));
	for (i = 0; i < parameter_c.out_feature_num; ++i) {
		out_feature_c.features[i] = t0 + i * in_feature_c.node_num;
	}
	// out_feature_c.features = (float**) malloc (parameter_c.out_feature_num * sizeof(float*));
	// for (i = 0; i < parameter_c.out_feature_num; ++i) {
	// 	out_feature_c.features[i] = (float*) malloc (in_feature_c.node_num * sizeof(float));
	// }

	for (i = 0; i < in_feature_c.node_num; ++i) {
		printf("\r%.2f%% Completed!", (float)i * 100.00 / (float)in_feature_c.node_num);
	    fflush(stdout);
		for (j = 0; j < parameter_c.out_feature_num; ++j) {
			
			out_feature_c.features[j][i] = parameter_c.biasses[j];
			for (k = 0; k < parameter_c.in_feature_num; ++k) {
				out_feature_c.features[j][i] += in_feature_c.features[k][i] * parameter_c.weights[k][j];
			}
			if(relu)
				out_feature_c.features[j][i] = MAX(0.00000, out_feature_c.features[j][i]);
		}
	}
	printf("\r\t\t\t\t\t\r");

	return out_feature_c;
}

void analyzer (feature_t feature_c, label_t label_c) {
	int i, j;
	int correct_num = 0;

	for (i = 0; i < feature_c.node_num; ++i) {
		float max_feature = feature_c.features[0][i];
		int max_idx = 0;
		for (j = 1; j < feature_c.feature_num; ++j) {
			if(feature_c.features[j][i] > max_feature) {
				max_feature = feature_c.features[j][i];
				max_idx = j;
			}
		}
		if (max_idx == label_c[i]) {
			correct_num++;
		}
	}
	
	printf("Accuracy: %.2f%%\n", (float)correct_num * 100.00 / (float)feature_c.node_num);
}