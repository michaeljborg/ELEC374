// Michael Borg - 20290079
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



int CoresPerSM(int major, int minor) {
	if (major == 8 && minor == 6) return 128; // Ampere achritecture for NVIDIA GeForce RTX 3060 Ti
	else return -1;
}


int main(int agrc, char* argvp[])
{
	int deviceCount = 0;
	char name[256];

	cudaGetDeviceCount(&deviceCount);

	printf("Device count: %d\n", deviceCount);
	for (int d = 0; d < deviceCount; d++)
	{
		cudaDeviceProp dp;
		cudaGetDeviceProperties(&dp, d);
		printf("Device %d: %s\n", d, dp.name);
		printf(" Clock Rate: %d KHz\n", dp.clockRate);
		printf(" Number of SMs: %d\n", dp.multiProcessorCount);
		printf(" Cores per SM: %d\n", CoresPerSM(dp.major, dp.minor));
		printf(" Total Cores: %d\n", dp.multiProcessorCount * CoresPerSM(dp.major, dp.minor));
		printf(" Warp Size: %d\n", dp.warpSize);
		printf(" Global Memory: %lu bytes\n", dp.totalGlobalMem);
		printf(" Constant Memory: %lu bytes\n", dp.totalConstMem);
		printf(" Shared Memory per Block: %lu bytes\n", dp.sharedMemPerBlock);
		printf(" Registers Available per Block: %d\n", dp.regsPerBlock);
		printf(" Max Threads per Block: %d\n", dp.maxThreadsPerBlock);
		printf(" Max Dimensions of a Block: %d x %d x %d\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
		printf(" Max Dimensions of a Grid: %d x %d x %d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
		printf("\n");
	}
	
	return 0;
}