# GCN
Graph Covolutional Network C implementation

1) Clone the repository: (clone it in greatlakes or else you have to copy the GCN folder containing the test data)
'git clone -b vojha1-eecs598 https://github.com/vojha1/eecs598.git'

2) The starting CUDA file is in project.cu

3) Build the executable code:
'nvcc project.cu utilities.c kernels.c'

4) Run the GCN:
'./a.out cora #citeseer, reddit'
