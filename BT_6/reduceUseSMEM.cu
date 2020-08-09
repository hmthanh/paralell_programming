%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;

#define N (1<<24)
#define BLOCK_DIM 256
#define RANDOM_MAX 100

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

double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(int *data, int size)
{
    srand(0);
    for (int i = 0; i < size; i++)
    {
        data[i] = (int)(rand()) / RANDOM_MAX;
    }
}

int reduceOnHost(int *data, int const size){
    if (size == 1)
        return data[0];
 
    if (size % 2 == 1) {
        data[0] += data[size];
    }
 
    int const stride = size / 2;
    for (int i = 0; i < stride; i++){
        data[i] += data[i + stride];
    }
    
    return reduceOnHost(data, stride);
}

__global__ void reduceUseSMEM(int *in, int *out, int n){
    // Each block loads data from GMEM to SMEM
    __shared__ int blkData[2 * 256];
    int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
    blkData[threadIdx.x] = in[numElemsBeforeBlk + threadIdx.x];
    blkData[blockDim.x + threadIdx.x] = in[numElemsBeforeBlk + blockDim.x + threadIdx.x];
    __syncthreads();

    // Each block does reduction with data on SMEM
    for (int stride = blockDim.x; stride > 0; stride /= 2){
        if (threadIdx.x < stride){
            blkData[threadIdx.x] += blkData[threadIdx.x + stride];
        }
        __syncthreads(); // Synchronize within threadblock
    }

    // Each block writes result from SMEM to GMEM
    if (threadIdx.x == 0)
        out[blockIdx.x] = blkData[0];
}

int main()
{
    int *in;
    int hostRes = 0, deviceRes = 0;
    size_t nBytes = N * sizeof(int);
    in = (int *)malloc(nBytes);
    initialData(in, N);

    dim3 blockSize(BLOCK_DIM);
    dim3 gridSize((N - 1) / (blockSize.x * 2) + 1);
    
    // malloc device global memory
    int *d_in, *d_out, *out;
    out = (int *)malloc(gridSize.x * sizeof(int));
    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_out, gridSize.x * sizeof(int)));
    CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));
 
    double iStart, iElaps;
    iStart = seconds();
    hostRes = reduceOnHost(in, N);
    iElaps = seconds() - iStart;
    printf("reduceOnHost : %f sec\n", iElaps);

    // #########################################
    iStart = seconds();
    reduceUseSMEM<<<gridSize, blockSize>>>(d_in, d_out, N);
    CHECK(cudaMemcpy(out, d_out, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < gridSize.x; i++){
        deviceRes += out[i];
    }
    iElaps = seconds() - iStart;
    printf("reduceUseSMEM : %f sec\n", iElaps);
    if (hostRes != deviceRes){
        printf("%d != %d", hostRes, deviceRes);
    }else{
        printf("%d == %d", hostRes, deviceRes);
    }
 
    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);

    free(in);
    free(out);

    return 0;
}