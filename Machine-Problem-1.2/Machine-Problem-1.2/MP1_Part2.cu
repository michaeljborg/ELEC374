// Michael Borg - 20290079
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>


__global__ void matrixMulGPU(float* P, float* M, float* N, int width) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx < width && ty < width) {
        float value = 0;
        for (int k = 0; k < width; k++) {
            float m = M[ty * width + k];
            float n = N[k * width + tx];
            value += m * n;
        }
        P[ty * width + tx] = value;
    }
}

void matrixMulCPU(float* P, const float* M, const float* N, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int n = 0; n < width; n++) {
                sum += M[row * width + n] * N[n * width + col];
            }
            P[row * width + col] = sum;
        }
    }
}

void matrixInit(float* data, int width) {
	for (int i = 0; i < width * width; i++) {
		data[i] = rand() / (float)RAND_MAX;
	}
}


int main() {
    // 100x100, 250x250, 500x500, 1000x100, 2500x2500
    int width = 0;
    for (int i = 0; i < 5; i++) {
        if (i == 0) {
            width = 100;
        } else if (i == 1) {
            width = 250;
        } else if (i == 2) {
            width = 500;
        } else if (i == 3) {
            width = 1000;
        } else if (i == 4) {
            width = 2500;
        }

        printf("%dx%d\n", width, width);
        printf("----------------------------------------------------------------------------------------\n");
        size_t size = width * width * sizeof(float);
        float* h_M, * h_N, * h_P; // Host memory
        float* d_M, * d_N, * d_P; // Device memory

        // Cuda timing event setup
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;


        // Allocate host memory
        h_M = (float*)malloc(size);
        h_N = (float*)malloc(size);
        h_P = (float*)malloc(size);

        // Initialize matrices M and N
        matrixInit(h_M, width);
        matrixInit(h_N, width);

        // Allocate device memory
        cudaMalloc((void**)&d_M, size);
        cudaMalloc((void**)&d_N, size);
        cudaMalloc((void**)&d_P, size);

        //---------------------------------------TEST 1-------------------------------------------\\

        cudaEventRecord(start);

        // Copy inputs to device (Host --> Device)
        cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Host to Device transfer time for size %d x %d: %f ms\n", width, width, milliseconds);

      
        cudaEventRecord(start);

        // Copy result back to host (Device -> Host)
        cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Device to Host transfer time for size %d x %d: %f ms\n", width, width, milliseconds);
        printf("----------------------------------------------------------------------------------------\n");

        //---------------------------------------TEST 1-------------------------------------------\\
        
 
 
        //---------------------------------------TEST 2-------------------------------------------\\
        
        int block_width = 0;
        for (int i = 0; i < 5; i++) {
            if (i == 0) {
                block_width = 2;
            }
            else if (i == 1) {
                block_width = 5;
            }
            else if (i == 2) {
                block_width = 10;
            }
            else if (i == 3) {
                block_width = 25;
            }
            else if (i == 4) {
                block_width = 32;
            }

            // Kernel launch parameters
            dim3 threadsPerBlock(block_width, block_width);
            dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

            // GPU START
            cudaEventRecord(start);

            // Launch the kernel for GPU matrix multiplication
            matrixMulGPU << <blocksPerGrid, threadsPerBlock >> > (d_P, d_M, d_N, width);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&milliseconds, start, stop);

            printf("GPU Matrix Multiplication time for size %d x %d and block width: %d: %f ms\n", width, width, block_width, milliseconds);

            // CPU START
            cudaEventRecord(start);

            matrixMulCPU(h_P, h_M, h_N, width);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&milliseconds, start, stop);

            printf("CPU Matrix Multiplication time for size %d x %d and block width %d: %f ms\n", width, width, block_width, milliseconds);

            bool correct = true;
            for (int i = 0; i < width * width; i++) {
                if (fabs(h_P[i] - h_P[i]) > 1e-5) {
                    correct = false;
                    break;
                }
            }

            // COMPARE RESULTS
            if (correct) {
                printf("Test PASSED\n\n");
            }
            else {
                printf("Test FAILED\n\n");
            }
            printf("----------------------------------------------------------------------------------------\n");
        }

        //---------------------------------------TEST 2-------------------------------------------\\



        // Free device memory
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);

        // Free host memory
        free(h_M);
        free(h_N);
        free(h_P);
 

    }
    return 0;
}