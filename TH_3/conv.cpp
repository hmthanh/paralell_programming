%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;

#define FILTER_SIZE 3
#define WIDTH (1<<10)
#define HEIGHT (1<<5)
__constant__ float d_flt[FILTER_SIZE];

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
        arr[i] = (float)(i);
    }
}

// ############### Host(CPU) ###############
void convOnHost(float *in, float *flt, float *out) {
    int filterPadding = FILTER_SIZE / 2;
    for (int i = 0; i < WIDTH; i++) {
        float s = 0;

        for (int j = 0; j < FILTER_SIZE; j++) {
            int index = min(max(0, i + j - filterPadding), WIDTH - 1);
            s += in[index];// * flt[j];
        }
        out[i] = s;
    }
}

__global__ void convUseDevice(float *d_in, float *d_out) {
    // for (int i = 0; i < WIDTH; i++){
    //     printf("%f ", d_in[i]);
    // }    
    extern __shared__ float ds_in[];
    int numElemsBeforeBlk = blockIdx.x * blockDim.x; // First elm of each block
    int idx = numElemsBeforeBlk + threadIdx.x;
    int filterPadding = FILTER_SIZE / 2;

    // FIRST BLOCK : SMEM range : (0, filterPadding) = d_in[0]
    if (threadIdx.x == 0){
        for (int iS = 0; iS <= filterPadding; iS++) {
            int sCol = max(idx - (filterPadding - iS), 0);
            ds_in[iS] = d_in[sCol];
        }
    }
    else if (threadIdx.x == blockDim.x - 1){
        for (int iS = 0; iS <= filterPadding; iS++) {
            int sCol = min(idx + iS, WIDTH - 1);
            ds_in[threadIdx.x + iS + filterPadding] = d_in[sCol];
        }
    }else{
        // Clone by copy
        ds_in[threadIdx.x + filterPadding] = d_in[idx];
    }
   
    __syncthreads();
    // d_out[idx] = ds_in[threadIdx.x + filterPadding];
    // d_out[idx] = ds_in[threadIdx.x];

    float s = 0;
    for (int j = 0; j < FILTER_SIZE; j++) {
        int index = threadIdx.x + j;
        s += ds_in[index];// * d_flt[j];
    }
    d_out[idx] = s;
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
    in = (float *)malloc(WIDTH * sizeof(float));
    flt = (float *)malloc(FILTER_SIZE * sizeof(float));
    hostRes = (float *)malloc(WIDTH * sizeof(float));
    deviceRes = (float *)malloc(WIDTH * sizeof(float));
    initialArray(in, WIDTH);
    initialArray(flt, FILTER_SIZE);

    // Allocate device memories
    float *d_in, *d_out;
    cudaMalloc(&d_in, WIDTH * sizeof(float));
    cudaMalloc(&d_out, WIDTH * sizeof(float));
    cudaMemcpy(d_in, in, WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_flt, flt, FILTER_SIZE * sizeof(float));


    // Launch the kernel
    dim3 blockSize(256);
    dim3 gridSize((WIDTH - 1) / blockSize.x + 1);
    double iStart, iElaps;

    // ##############################################
    // convOnHost
    iStart = seconds();
    convOnHost(in, flt, hostRes);
    iElaps = seconds() - iStart;
    printf("convOnHost : %f sec\n", iElaps);

    // ##############################################
    // convUseRMEMAndGMEM
    iStart = seconds();
    size_t sBytes = (blockSize.x + (FILTER_SIZE>>1)<<1)* sizeof(float);
    // int floatSize = blockSize.x + ((FILTER_SIZE >> 1) << 1);
    printf("<<<%d, %d>>>", blockSize.x, gridSize.x);

    convUseDevice<<<gridSize, blockSize, sBytes>>>(d_in, d_out);
    iElaps = seconds() - iStart;
    printf("convUseRMEMAndGMEM : %f sec\n", iElaps);
    cudaMemcpy(deviceRes, d_out, WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Host :");
    for (int i = 0; i< WIDTH; i++) {
        printf("%d[%f] ", i, hostRes[i]);
    }
    printf("\nDevice :");
    for (int i = 0; i< WIDTH; i++) {
        printf("%d[%f] ", i, deviceRes[i]);
    }
    printf("\n");
    checkResult(hostRes, deviceRes, WIDTH);

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