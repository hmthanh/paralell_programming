%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;

#define RANDOM_MAX 10
#define BLOCK_SIZE 512

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

void initialVector(int *vector, int size)
{
    srand(0);
    for (int i = 0; i < size; i++)
    {
        vector[i] = (int)(rand()) / RANDOM_MAX;
    }
}

void sumOnHost(int *in1, int *in2, int *out, int size){
    for (int i = 0; i < size; i++){
        out[i] = (in1[i] + in2[i])/2;
    }
}

// ############### Device(CPU) ###############
// Hàm thực hiện reduce trên CPU
__global__ void sumOnDevice(int *in1, int *in2, int *out, int size){
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < size){
        out[idx] = (in1[idx] + in2[idx])/2;
    }
}
bool checkResult(int *hostRef, int *gpuRef, unsigned int size)
{
    double epsilon = 1.0E-8;
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
    printf("############ GPU Properties ############\n");
    // Chọn GPU thực thi câu lệnh    
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Khởi tạo kích thước vector
    unsigned int size = 1 << 27; // 2^24 - Đổi số nhỏ hơn để tính toán nhanh hơn
    int *A, *B, *hostRef, *gpuRef;
    size_t nBytes = size * sizeof(int);
    A = (int *) malloc(nBytes);
    B = (int *) malloc(nBytes);
    hostRef = (int *) malloc(nBytes);
    gpuRef = (int *) malloc(nBytes);
    
    initialVector(A, size);
    initialVector(B, size);
    printf("Kích thước mảng : %d\n", size);

    // Biến tính thời gian chạy
    double iStart, iElaps;

    // Kernel được cấu hình với 1D grid và 1D blocks
    dim3 blockSize (BLOCK_SIZE);
    dim3 gridSize  ((size - 1) / blockSize.x + 1);
    printf("Kích thước : <<<Grid (%d, %d), Block (%d, %d)>>>\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);
    
    // CPU ##########################################################
    printf("ID| Kernel\t\t\t\t|Time \t\t| Sum result \n");
    // ############ 1. sumOnHost #############
    iStart = seconds();
    sumOnHost(A, B, hostRef, size);
    iElaps = seconds() - iStart;
    printf("1 | sumOnHost \t\t\t\t| %f sec\t\n", iElaps);

    // GPU ##########################################################
    // ############ 2. sumOnDeviceWithoutStream #############
    // Cấp phát bộ nhớ trên device (GPU)
    int *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK(cudaMalloc((int**)&d_A, nBytes));
    CHECK(cudaMalloc((int**)&d_B, nBytes));
    CHECK(cudaMalloc((int**)&d_C, nBytes));
    
    iStart = seconds();

    // Copy inputs to device
    cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, nBytes, cudaMemcpyHostToDevice);

    sumOnDevice<<<gridSize, blockSize, 0, 0>>>(d_A, d_B, d_C, size);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    iElaps = seconds() - iStart;
    int isTrue = checkResult(hostRef, gpuRef, size);
    printf("2 | sumOnDeviceWithoutStream \t\t| %f sec\t| %d\t\n", iElaps, isTrue);

    // ############ 3. sumOnDevice2Stream #############
    unsigned int nStream = 2;

    int *h_A, *h_B, *h_hostRef, *h_gpuRef;
    cudaHostAlloc(&h_A, nBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_B, nBytes, cudaHostAllocDefault);
    initialVector(h_A, size);
    initialVector(h_B, size);
    cudaHostAlloc(&h_hostRef, nBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_gpuRef, nBytes, cudaHostAllocDefault);
    memset(h_hostRef, 0, nBytes);
    memset(h_gpuRef,  0, nBytes);

    // Tạo stream
    cudaStream_t stream1[nStream];
    for (int i = 0; i < nStream; i++)
        cudaStreamCreate(&stream1[i]);

    int iSize = size/nStream;
    int iBytes = iSize * sizeof(int);
    
    iStart = seconds();

    for (int i = 0; i < nStream; ++i)
    {
        int ioffset = i * iSize;
        CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream1[i]));
        CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream1[i]));
        sumOnDevice<<<gridSize, blockSize, 0, stream1[i]>>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iSize);
        CHECK(cudaMemcpyAsync(&h_gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream1[i]));
    }

    iElaps = seconds() - iStart;
    isTrue = checkResult(hostRef, gpuRef, size);
    printf("3 | sumOnDevice2Stream \t\t\t| %f sec\t| %d\t\n", iElaps, isTrue);
    for (int i = 0; i <nStream; i++)
        cudaStreamDestroy(stream1[i]);

    // ############ 4. sumOnDevice3Stream #############
    nStream = 3;

    cudaHostAlloc(&h_A, nBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_B, nBytes, cudaHostAllocDefault);
    initialVector(h_A, size);
    initialVector(h_B, size);
    cudaHostAlloc(&h_hostRef, nBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_gpuRef, nBytes, cudaHostAllocDefault);
    memset(h_hostRef, 0, nBytes);
    memset(h_gpuRef,  0, nBytes);

    // Tạo stream
    cudaStream_t stream2[nStream];
    for (int i = 0; i < nStream; i++)
        cudaStreamCreate(&stream2[i]);

    iSize = size/nStream;
    iBytes = iSize * sizeof(int);
    
    iStart = seconds();
    
    for (int i = 0; i < nStream; ++i){
        int ioffset = i * iSize;
        CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream2[i]));
        CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream2[i]));
        sumOnDevice<<<gridSize, blockSize, 0, stream2[i]>>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iSize);
        CHECK(cudaMemcpyAsync(&h_gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream2[i]));
    }

    iElaps = seconds() - iStart;
    isTrue = checkResult(hostRef, gpuRef, size);
    printf("4 | sumOnDevice3Stream \t\t\t| %f sec\t| %d\t\n", iElaps, isTrue);
    for (int i = 0; i <nStream; i++)
        cudaStreamDestroy(stream2[i]);

    // ############ 5. sumOnDevice3StreamUseEvent #############
    // nStream = 3;
    // Tạo stream
    cudaStream_t stream3[nStream];
    for (int i = 0; i < nStream; i++)
        cudaStreamCreate(&stream3[i]);

    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *) malloc(nStream * sizeof(cudaEvent_t));

    for (int i = 0; i < nStream; i++){
        CHECK(cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
    }

    iSize = size/nStream;
    iBytes = iSize * sizeof(int);
    
    iStart = seconds();
    
    for (int i = 0; i < nStream; ++i){
        int ioffset = i * iSize;
        CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream3[i]));
        CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream3[i]));
        sumOnDevice<<<gridSize, blockSize, 0, stream3[i]>>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iSize);
        CHECK(cudaMemcpyAsync(&h_gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream3[i]));
        CHECK(cudaEventRecord(kernelEvent[i], stream3[i]));
        CHECK(cudaStreamWaitEvent(stream3[nStream - 1], kernelEvent[i], 0));
    }

    iElaps = seconds() - iStart;
    isTrue = checkResult(hostRef, gpuRef, size);
    printf("5 | sumOnDevice3StreamUseEvent \t\t| %f sec\t| %d\t\n", iElaps, isTrue);
    for (int i = 0; i <nStream; i++){
        CHECK(cudaStreamDestroy(stream3[i]));
        CHECK(cudaEventDestroy(kernelEvent[i]));
    }
  
    // free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free pinned memory
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(h_hostRef));
    CHECK(cudaFreeHost(h_gpuRef));

    // free host memory
    free(A);
    free(B);
    free(hostRef);
    free(gpuRef);
    free(kernelEvent);

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}