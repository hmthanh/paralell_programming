#include <stdio.h>
#include <stdlib.h>
#include <chrono>

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

void addMatOnHost(float *in1, float *in2, float *out,
                  int nx, int ny)
{
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            int idx = i * nx + j;
            out[idx] = in1[idx] + in2[idx];
        }
    }
}

void printMatrix(float *matrix, int nx, int ny)
{
    printf("\n");
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            int idx = i * ny + j;
            printf("%f ", matrix[idx]);
        }
        printf("\n");
    }
}

int main()
{
    int nx, ny;         // Số cột và số dòng
    float *in1, *in2; // input matrix
    float *out;         // output vector

    nx = 3;
    ny = 3;

    int size = nx * ny * sizeof(float);

    in1 = (float *)malloc(size);
    in2 = (float *)malloc(size);
    out = (float *)malloc(size);

    // Setup input values
    srand(time(0));
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            int idx = i * ny + j;
            in1[idx] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            in2[idx] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }

    // Allocate vector to device memory
    float *d_in1, *d_in2, *d_out;
    cudaMalloc(&d_in1, size);
    cudaMalloc(&d_in2, size);
    cudaMalloc(&d_out, size);

    // Copy inputs to device
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

    printf("Input 1 : ");
    printMatrix(in1, nx, ny);
 
    printf("Input 2 : ");
    printMatrix(in2, nx, ny);
 

    // // Launch add() kernel on GPU
    dim3 blockSize(32, 32);
    dim3 gridSize((nx - 1) / blockSize.x + 1, (ny - 1) / blockSize.y + 1);

    auto start_host = high_resolution_clock::now();
    addMatOnHost(in1, in2, out, nx, ny);
    auto stop_host = high_resolution_clock::now();
    auto duration_host = duration_cast<microseconds>(stop_host - start_host);
    printf("Time host : %d milliseconds\n", duration_host.count());

    auto start_device = high_resolution_clock::now();
    addMatOnDevice2D<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, nx, ny);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    auto stop_device = high_resolution_clock::now();
    auto duration_device = duration_cast<microseconds>(stop_device - start_device);
    printf("Time device : %d milliseconds\n", duration_device.count());

    printf("Output : ");
    printMatrix(out, nx, ny);

    // Cleanup
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);

    free(in1);
    free(in2);
    free(out);

    return 0;
}