// Michael Borg - 20290079
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>



// Part 2 Matrix Multiplication that can use non matching matrix and tile sizes
___global__ void TiledMatrixMulKernel(float* P, float* M, float* N, int MHeight, int MWidth, int NWidth, int TileHeight, int TileWidth) {
    extern __shared__ float sharedMem[];

    int sharedMemSize = TileHeight * TileWidth;
    float* Ms = sharedMem;
    float* Ns = &sharedMem[sharedMemSize]; 
    

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TileHeight + ty; // Use TileHeight for y dimension
    int Col = bx * TileWidth + tx; // Use TileWidth for x dimension

    float Pvalue = 0;
    int numTilesAcrossM = (MWidth + TileWidth - 1) / TileWidth; // Tiles needed to cover the width of M (and height of N)

    for (int m = 0; m < numTilesAcrossM; ++m) {
        int mCol = m * TileWidth + tx;
        int nRow = m * TileWidth + ty; // Note: nRow traverses MWidth, so we still use TileWidth for calculation

        // Load M tile
        if (Row < MHeight && mCol < MWidth)
            Ms[ty * TileWidth + tx] = M[Row * MWidth + mCol];
        else
            Ms[ty * TileWidth + tx] = 0.0;

        // Load N tile
        if (Col < NWidth && nRow < MWidth)
            Ns[ty * TileWidth + tx] = N[nRow * NWidth + Col];
        else
            Ns[ty * TileWidth + tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TileWidth; ++k) { // Iterate across the width of the tile
            Pvalue += Ms[ty * TileWidth + k] * Ns[k * TileWidth + tx];
        }

        __syncthreads();
    }

    if (Row < MHeight && Col < NWidth) {
        P[Row * NWidth + Col] = Pvalue;
    }
}


// updated CPU matrix mul for non matching sizes
void matrixMulCPU(float* P, const float* M, const float* N, int M_height, int M_width, int N_width) {
    for (int row = 0; row < M_height; row++) {
        for (int col = 0; col < N_width; col++) {
            float sum = 0.0f;
            for (int n = 0; n < M_width; n++) { // Loop over the width of M, which is also the height of N
                sum += M[row * M_width + n] * N[n * N_width + col];
            }
            P[row * N_width + col] = sum;
        }
    }
}


// updated Init for non matching sizes
void matrixInit(float* data, int height, int width) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            data[row * width + col] = rand() / (float)RAND_MAX;
        }
    }
}


int main() {

    int numExperiments = 10;

    // Test 1: M = 400x450, N = 450x500
    // Test 2: M = 1200x1350, N = 1350x1150
    // Tile Size: 9x16

    int M_height = 400;
    int M_width = 450;
    int N_height = 450;
    int N_width = 500;
    int tile_height = 9;
    int tile_width = 16;

    // Sizes
    size_t M_size = M_height * M_width * sizeof(float);
    size_t N_size = N_height * N_width * sizeof(float);
    size_t P_size = M_height * N_width * sizeof(float);

    float* h_M, * h_N, * h_P, * h_P_cpu; // Host memory
    float* d_M, * d_N, * d_P; // Device memory

    // Allocate host memory
    h_M = (float*)malloc(M_size);
    h_N = (float*)malloc(N_size);
    h_P = (float*)malloc(P_size);
    h_P_cpu = (float*)malloc(P_size);

    // Initialize matrices M and N
    matrixInit(h_M, M_height, M_width);
    matrixInit(h_N, N_height, N_width);


    // Allocate device memory
    cudaMalloc((void**)&d_M, M_size);
    cudaMalloc((void**)&d_N, N_size);
    cudaMalloc((void**)&d_P, P_size);


    // Copy inputs to device (Host --> Device)
    cudaMemcpy(d_M, h_M, M_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, N_size, cudaMemcpyHostToDevice);



    float totalTime = 0.0f; // Time for all experiments

    matrixMulCPU(h_P_cpu, h_M, h_N, M_height, M_width, N_width);


    // Cuda timing event setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Matrix Size: %dx%d and %dx%d, Tile Size: %dx%d\n", M_height, M_width, N_height, N_width, tile_width, tile_height);

    for (int k - 0; k < numExperiments; k++) {
        printf("Experiment %d: ", k+1);

        dim3 dimBlock(tile_height, tile_width);
        dim3 dimGrid((N_width + tile_width - 1) / tile_width, (M_height + tile_height - 1) / tile_height);

        cudaEventRecord(start);
        int sharedMemSize = 2 * tile_height * tile_width * sizeof(float);

        TiledMatrixMulKernel<<<dimGrid, dimBlock, sharedMemSize>>>(d_P, d_M, d_N, M_height, M_width, N_width, tile_height, tile_width);


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime =  totalTime + milliseconds;

        cudaMemcpy(h_P, d_P, P_size, cudaMemcpyDeviceToHost);



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
        
    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_P_cpu);

    return 0;
}