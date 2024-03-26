// Michael Borg - 20290079
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>


__global__ void TiledMatrixMulKernel(float* P, float* M, float* N, int Width, int TileWidth) {
    extern __shared__ float sharedMem[]; // Use extern to specify dynamic shared memory

    // Calculate the total size needed for Ms and Ns within the shared memory
    // Each matrix (Ms and Ns) needs TileWidth * TileWidth space
    int sharedMemSizePerMatrix = TileWidth * TileWidth;

    // Cast the shared memory to float and split it for Ms and Ns
    float* Ms = sharedMem;
    float* Ns = &sharedMem[sharedMemSizePerMatrix];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TileWidth + ty;
    int Col = bx * TileWidth + tx;

    float Pvalue = 0;

    for (int m = 0; m < (Width + TileWidth - 1) / TileWidth; ++m) {
        int mIndex = Row * Width + (m * TileWidth + tx);
        int nIndex = (m * TileWidth + ty) * Width + Col;

        if (Row < Width && m * TileWidth + tx < Width)
            Ms[ty * TileWidth + tx] = M[mIndex];
        else
            Ms[ty * TileWidth + tx] = 0.0;

        if (Col < Width && m * TileWidth + ty < Width)
            Ns[ty * TileWidth + tx] = N[nIndex];
        else
            Ns[ty * TileWidth + tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TileWidth; ++k) {
            Pvalue += Ms[ty * TileWidth + k] * Ns[k * TileWidth + tx];
        }

        __syncthreads();
    }

    if (Row < Width && Col < Width) {
        P[Row * Width + Col] = Pvalue;
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
    // 100x100, 250x250, 500x500, 1000x100, 1500x2500
    int sizes[] = {100, 250, 500, 1000, 1500};
    int tileWidths[] = {2, 5, 10, 25};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    int numTileWidths = sizeof(tileWidths) / sizeof(tileWidths[0]);
    int numExperiments = 10;
    

    for (int i = 0; i < numSizes; i++) {
        int width = sizes[i];
        size_t size = width * width * sizeof(float);
        int block = 0;
        

        float* h_M, * h_N, * h_P, * h_P_cpu; // Host memory
        float* d_M, * d_N, * d_P; // Device memory

        // Allocate host memory
        h_M = (float*)malloc(size);
        h_N = (float*)malloc(size);
        h_P = (float*)malloc(size);
        h_P_cpu = (float*)malloc(size);

        // Initialize matrices M and N
        matrixInit(h_M, width);
        matrixInit(h_N, width);



        // Allocate device memory
        cudaMalloc((void**)&d_M, size);
        cudaMalloc((void**)&d_N, size);
        cudaMalloc((void**)&d_P, size);

        // Copy inputs to device (Host --> Device)
        cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);


        for (int j = 0; j < numTileWidths; j++) {
            int tileWidth = tileWidths[j];
            float totalTime = 0.0f; // Time for all experiments

            if (block == 0) {
                matrixMulCPU(h_P_cpu, h_M, h_N, width);
                block = 1;
            }
            
            // Cuda timing event setup
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            printf("Matrix Size: %dx%d, Tile Width: %d\n", width, width, tileWidth);


            for (int k = 0; k < numExperiments; k++) {



                printf("Experiment %d: ", k+1);
            
                dim3 dimBlock(tileWidth, tileWidth);
                dim3 dimGrid((width + tileWidth - 1) / tileWidth, (width + tileWidth - 1) / tileWidth);

                cudaEventRecord(start);

                int sharedMemSize = 2 * tileWidth * tileWidth * sizeof(float);
                TiledMatrixMulKernel << <dimGrid, dimBlock, sharedMemSize >> > (d_P, d_M, d_N, width, tileWidth);


                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                totalTime =  totalTime + milliseconds;

                cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);


                bool correct = true;
                for (int i = 0; i < width * width; i++) {
                    if (fabs(h_P_cpu[i] - h_P[i]) > 1e-5) {
                        correct = false;
                        break;
                    }
                }

                // COMPARE RESULTS
                if (correct) {
                    printf("Test PASSED -> ");
                    printf("Time: %f ms\n", milliseconds);
                }
                else {
                    printf("Test FAILED -> ");
                    printf("Time: %f ms\n", milliseconds);
                }


            }

            float averageTime = totalTime / numExperiments;
            printf("Total Time: %f ms \n", totalTime);
            printf("NumExperiments: %d\n", numExperiments);
            printf("Average Time: %f ms\n\n", averageTime);



            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
        }

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