%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;

#define NF 100
#define NI (1<<24)
#define NO (NI - NF + 1)
__constant__ float d_flt[NF];
__device__ float d_gflt[NF];

// ############### COMMON ###############
#define CHECK(call)                                                        \
{                                                                          \
    const cudaError_t error = call;                                        \
    if (error != cudaSuccess)                                              \
    {                                                                      \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                           \
    }                                                                      \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialArray(float *arr, int size)
{
    srand(0);
    for (int i = 0; i < size; i++)
    {
        arr[i] = (float)(rand())/RAND_MAX;
    }
}

// ############### Host(CPU) ###############
void convOnHost(float *in, float *flt, float *out){
    for (int i = 0; i < NO; i++){
        float s = 0;

        for (int j = 0; j < NF; j++){
            s += in[i + j] * flt[j];
        }
        out[i] = s;
    }
}

// ############### Device(GPU) ###############
__global__ void convUseGMEM(float *d_in, float *d_out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NO){
        d_out[i] = 0; // use all global memory
        for (int j = 0; j < NF; j++){
            d_out[i] += d_gflt[j] * d_in[i + j]; // d_gflt for GMEM
        }
    }
}

__global__ void convUseCMEM(float *d_in, float *d_out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NO){
        d_out[i] = 0;
        for (int j = 0; j < NF; j++){
            d_out[i] += d_flt[j] * d_in[i + j]; // d_flt for RMEM
        }
    }
}


__global__ void convUseRMEMAndGMEM(float *d_in, float *d_out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NO){
        float s = 0; // s for RMEM
        for (int j = 0; j < NF; j++){
            s += d_gflt[j] * d_in[i + j]; // d_gflt for GMEM
        }
        d_out[i] = s;
    }
}


bool checkResult(float *hostRef, float *gpuRef, unsigned int size)
{
    double epsilon = 1.0E-3;
    bool isTrue = 1;

    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
            isTrue = 0;
            return isTrue;
        }
    }

    return isTrue;
}

int main()
{
    // Set up data for input and filter
    float *in, *flt, *hostRes, *deviceRes;
    in = (float *) malloc(NI * sizeof(float));
    flt = (float *) malloc(NF * sizeof(float));
    hostRes = (float *) malloc(NO * sizeof(float));
    deviceRes = (float *) malloc(NO * sizeof(float));
    initialArray(in, NI);
    initialArray(flt, NF);

    // Allocate device memories
    float *d_in, *d_out;
    cudaMalloc(&d_in, NI * sizeof(float));
    cudaMalloc(&d_out, NO * sizeof(float));
    cudaMemcpy(d_in, in, NI * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_gflt, flt, NF * sizeof(float));
    cudaMemcpyToSymbol(d_flt, flt, NF * sizeof(float));


    // Launch the kernel
    dim3 blockSize(512);
    dim3 gridSize((NO - 1) / blockSize.x + 1);
    double iStart, iElaps;

    // ##############################################
    // convOnHost
    iStart = seconds();
    convOnHost(in, flt, hostRes);
    iElaps = seconds() - iStart;
    printf("convOnHost : %f sec\n", iElaps);

    // ##############################################
    // convUseGMEM
    iStart = seconds();
    convUseGMEM<<<gridSize, blockSize>>>(d_in, d_out);
    iElaps = seconds() - iStart;
    printf("convUseGMEM : %f sec\n", iElaps);
    // Copy results from device memory to host memory
    cudaMemcpy(deviceRes, d_out, NO * sizeof(float), cudaMemcpyDeviceToHost);
    checkResult(hostRes, deviceRes, NO);

    // ##############################################
    // convUseCMEM
    iStart = seconds();
    convUseCMEM<<<gridSize, blockSize>>>(d_in, d_out);
    iElaps = seconds() - iStart;
    printf("convUseCMEM : %f sec\n", iElaps);
    cudaMemcpy(deviceRes, d_out, NO * sizeof(float), cudaMemcpyDeviceToHost);
    checkResult(hostRes, deviceRes, NO);
    
    // ##############################################
    // convUseRMEMAndGMEM
    iStart = seconds();
    convUseRMEMAndGMEM<<<gridSize, blockSize>>>(d_in, d_out);
    iElaps = seconds() - iStart;
    printf("convUseRMEMAndGMEM : %f sec\n", iElaps);
    cudaMemcpy(deviceRes, d_out, NO * sizeof(float), cudaMemcpyDeviceToHost);
    checkResult(hostRes, deviceRes, NO);
    
    // Free device memories
    cudaFree(d_in);
    cudaFree(d_out);

    // free
    free(in);
    free(flt);
    free(hostRes);
    free(deviceRes);

    return 0;
}