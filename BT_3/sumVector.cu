%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;

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

void initialData(float *data, int size)
{
    srand(0);
    for (int i = 0; i < size; i++)
    {
        data[i] = (float)(rand()) / RAND_MAX;
    }
}

void printFirst10(float *data, int size){
    for (int i = 0; i < size && i < 10; i++){
        printf("%f\n", data[i]);
    }
}

float recursiveReduce(float *data, int const size)
{
    if (size == 1)
        return data[0];
 
    if (size % 2 == 1) {
        data[0] += data[size];
    }
 
    int const stride = size / 2;
    for (int i = 0; i < stride; i++){
        data[i] += data[i + stride];
    }
    
    return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(float *g_idata, float *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // convert global data pointer to the local pointer of this block
    float *idata = g_idata + blockIdx.x * blockDim.x;
 
    // boundary check
    if (idx >= n) return;

    // in-place reduction in gloabl memory
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        if ((tid % (2 * stride)) == 0){
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(float *g_idata, float *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in gloabl memory
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x){
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];

}

int main()
{
    // set up device
    int dev = 0;
    cudaSetDevice(dev);
    int const BLOCK_SIZE = 1024;

    // set up vector
    int size = (pow(2, 13) + 1) * (pow(2, 13) + 1);
    
    float *data, host_total, device_total = 0; // vector
    
    int nBytes = size * sizeof(float);
    data = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(data, size);
    printFirst10(data, size);

    // malloc device global memory
    float *g_idata, *g_odata;
    CHECK(cudaMalloc((float **)&g_idata, nBytes));
    CHECK(cudaMalloc((float **)&g_odata, nBytes));

    CHECK(cudaMemcpy(g_idata, data, nBytes, cudaMemcpyHostToDevice));
 
    double host_start = seconds();
    host_total = recursiveReduce(data, size);
    double host_elaps = seconds() - host_start;
    printf("Total host : %f, Time host : %f sec\n", host_total, host_elaps);

    // Launch add() kernel on GPU
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    unsigned int gridWith = (size + blockSize.x - 1) / blockSize.x;
    printf("gridWith %d\n", gridWith);
    dim3 gridSize(gridWith, 1, 1);

    // output each block in gpu
    float *odata;
    odata = (float *)malloc(gridWith * sizeof(float));

    double device_start = seconds();
    reduceNeighboredLess<<<gridSize, blockSize>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK(cudaMemcpy(odata, g_odata, gridWith * sizeof(float), cudaMemcpyDeviceToHost));
    printFirst10(odata, gridWith);
    for (int i = 0; i < gridWith; i++){
        device_total += odata[i];
    }
    double device_elaps = seconds() - device_start;
    printf("Total device %f, Time device : %f sec\n", device_total, device_elaps);
 
    // Cleanup
    cudaFree(g_idata);
    cudaFree(g_odata);

    free(data);
    free(odata);

    return 0;
}