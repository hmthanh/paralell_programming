#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <string>
#include <iostream>

using namespace std::chrono;
using namespace std;

__global__ void addMatOnDevice2D(float *in1, float *in2, float *out, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny)
    {
        int i = iy * nx + ix;
        out[i] = in1[i] + in2[i];
    }
}

__global__ void addMatOnDevice1D(float *in1, float *in2, float *out, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            out[idx] = in1[idx] + in2[idx];
        }
    }
}

__global__ void addMatOnDeviceMix(float *in1, float *in2, float *out, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = blockIdx.y;
    if (ix < nx)
    {
        int idx = iy * nx + ix;
        out[idx] = in1[idx] + in2[idx];
    }
}

void addMatOnHost(float *in1, float *in2, float *out,
                  int nx, int ny)
{
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++){
            int idx = i * nx + j;
            out[idx] = in1[idx] + in2[idx];
        }
}

void printMatrix(float *matrix, int nx, int ny)
{
    printf("\n");
    for (int i = 0; i < ny; i++){
        for (int j = 0; j < nx; j++){
            int idx = i * ny + j;
            printf("%f ", matrix[idx]);
        }
        printf("\n");
    }
}

void calcTimeOnDevice(float *in1, float *in2, float *out, int nx, int ny, dim3 blockSize, dim3 gridSize, int typeDevice)
{
    int size = nx * ny * sizeof(float);

    // Allocate vector to device memory
    float *d_in1, *d_in2, *d_out;
    cudaMalloc(&d_in1, size);
    cudaMalloc(&d_in2, size);
    cudaMalloc(&d_out, size);

    // Copy inputs to device
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

    auto start_device = high_resolution_clock::now();
    string deviceName = "";
    if (typeDevice == 1){
        deviceName = "addMatOnDevice2D";
        addMatOnDevice2D<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, nx, ny);
    }else if (typeDevice == 2){
        deviceName = "addMatOnDevice1D";
        addMatOnDevice1D<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, nx, ny);
    }else if (typeDevice == 3){
        deviceName = "addMatOnDevice2DNotMix";
        addMatOnDevice2D<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, nx, ny);
    }else {
        deviceName = "addMatOnDeviceMix";
        addMatOnDeviceMix<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, nx, ny);
    }
    
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    auto stop_device = high_resolution_clock::now();
    auto duration_device = duration_cast<microseconds>(stop_device - start_device);
    auto duration = duration_device.count();
 
    printf("%s|%d x %d\t|%d x %d\t|%d ms\t\n", deviceName.c_str(), blockSize.x, blockSize.y, gridSize.x, gridSize.y, duration);

     // Cleanup
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
}


int main()
{
    int nx, ny;       // Số cột và số dòng
    float *in1, *in2; // input matrix
    float *out;       // output vector

    nx = pow(2, 13) + 1;
    ny = pow(2, 13) + 1;

    int size = nx * ny * sizeof(float);

    in1 = (float *)malloc(size);
    in2 = (float *)malloc(size);
    out = (float *)malloc(size);

    // Setup input values
    srand(time(0));
    for (int i = 0; i < ny; i++){
        for (int j = 0; j < nx; j++){
            int idx = i * ny + j;
            in1[idx] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            in2[idx] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }

    auto start_host = high_resolution_clock::now();
    addMatOnHost(in1, in2, out, nx, ny);
    auto stop_host = high_resolution_clock::now();
    auto duration_host = duration_cast<microseconds>(stop_host - start_host);
    printf("Function\t|Block size\t|Grid size\t|Time (ms)\n");
    printf("addMatOnHost\t|\t\t|\t\t|%d\n", duration_host.count());


    /********************************
    addMatOnDevice2D
    *********************************/
    int typeDevice = 1;

    dim3 blockSize(32, 32);
    dim3 gridSize(257, 257);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    blockSize = dim3(16, 32);
    gridSize = dim3(513, 257);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    blockSize = dim3(32, 16);
    gridSize = dim3(257, 513);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    blockSize = dim3(16, 16);
    gridSize = dim3(513, 513);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
    /********************************
    addMatOnDevice1D
    *********************************/
    typeDevice = 2;

    blockSize = dim3(32, 1);
    gridSize = dim3(257, 1);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
 
    blockSize = dim3(64, 1);
    gridSize = dim3(129, 1);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
 
    blockSize = dim3(128, 1);
    gridSize = dim3(65, 1);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
 
    /********************************
    addMatOnDevice2DNotMix
    *********************************/
    typeDevice = 3;

    blockSize = dim3(32, 1);
    gridSize = dim3(257, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
 
    blockSize = dim3(64, 1);
    gridSize = dim3(129, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
 
    blockSize = dim3(128, 1);
    gridSize = dim3(65, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    blockSize = dim3(256, 1);
    gridSize = dim3(33, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    blockSize = dim3(512, 1);
    gridSize = dim3(17, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    blockSize = dim3(1024, 1);
    gridSize = dim3(9, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
 
    blockSize = dim3(2048, 1);
    gridSize = dim3(5, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    /********************************
    addMatOnDeviceMix
    *********************************/
    typeDevice = 4;

    blockSize = dim3(32, 1);
    gridSize = dim3(257, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
 
    blockSize = dim3(64, 1);
    gridSize = dim3(129, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
 
    blockSize = dim3(128, 1);
    gridSize = dim3(65, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    blockSize = dim3(256, 1);
    gridSize = dim3(33, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    blockSize = dim3(512, 1);
    gridSize = dim3(17, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);

    blockSize = dim3(1024, 1);
    gridSize = dim3(9, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
 
    blockSize = dim3(2048, 1);
    gridSize = dim3(5, 8193);
    calcTimeOnDevice(in1, in2, out, nx, ny, blockSize, gridSize, typeDevice);
    
    free(in1);
    free(in2);
    free(out);

    return 0;
}