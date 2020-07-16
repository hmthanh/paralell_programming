#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;

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

void initialData(double *data, int size)
// Khởi tạo vector ban đầu có kích thước size, kiểu double và giá random trong khoảng [0, 1]
{
    srand(0);
    for (int i = 0; i < size; i++)
    {
        data[i] = (double)(rand()) / RAND_MAX;
    }
}

double sumGPU(double *data, int size){
    double sum = 0;
    for(int i = 0; i < size; i++){
        sum += data[i];
    }

    return sum;
}

// ############### Device(CPU) ###############
// Hàm thực hiện reduce trên CPU
double recursiveReduce(double *data, int const size)
{
    if (size == 1) return data[0];
  
    int const stride = size / 2;
    for (int i = 0; i < stride; i++){
        data[i] += data[i + stride];
    }
    
    return recursiveReduce(data, stride);
}

// ############### Device(GPU) ###############
// Neighbored Pair phân kỳ
__global__ void reduceNeighbored (double *g_idata, double *g_odata,
    unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Chuyển từ con trỏ toàn cục sang con trỏ của block này
    double *idata = g_idata + blockIdx.x * blockDim.x;

    // Kiểm tra nếu vượt qua kích thước mảng
    if (idx >= n) return;

    // Thực hiện tính tổng ở bộ nhớ toàn cục
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // Đồng bộ hóa trong một threadBlock
        __syncthreads();
    }

    // Ghi kết quả cho block này vào bộ nhớ toàn cục
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Neighbored Pair cài đặt với ít phân kỳ bằng cách thực thi trong một block
__global__ void reduceNeighboredLess (double *g_idata, double *g_odata,
    unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Chuyển từ con trỏ toàn cục sang con trỏ của block này
    double *idata = g_idata + blockIdx.x * blockDim.x;

    // Kiểm tra nếu vượt qua kích thước mảng
    if(idx >= n) return;

    // Thực hiện tính tổng ở bộ nhớ toàn cục
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // Chuyển tid sang bộ nhớ của một block (register - thanh ghi)
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }

        // Đồng bộ hóa trong một threadBlock
        __syncthreads();
    }

    // Ghi kết quả cho block này vào bộ nhớ toàn cục
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved (double *g_idata, double *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Chuyển từ con trỏ toàn cục sang con trỏ của block này
    double *idata = g_idata + blockIdx.x * blockDim.x;

    // Kiểm tra nếu vượt qua kích thước mảng
    if(idx >= n) return;

    // Thực hiện tính tổng ở bộ nhớ toàn cục
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // Ghi kết quả cho block này vào bộ nhớ toàn cục
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2 (double *g_idata, double *g_odata,
    unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Chuyển từ con trỏ toàn cục sang con trỏ của block này
    double *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2 data blocks
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    // Đồng bộ hóa các group data trong 2 thread kết cận
    __syncthreads();

    // Thực hiện tính tổng ở bộ nhớ toàn cục
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // Đồng bộ hóa trong một threadBlock
        __syncthreads();
    }

    // Ghi kết quả cho block này vào bộ nhớ toàn cục
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(double *g_idata, double *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // Chuyển từ con trỏ toàn cục sang con trỏ của block này
    double *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n)
    {
        double a1 = g_idata[idx];
        double a2 = g_idata[idx + blockDim.x];
        double a3 = g_idata[idx + 2 * blockDim.x];
        double a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }
    __syncthreads(); // Đồng bộ hóa các group data trong 4 thread kết cận

    // Thực hiện tính tổng ở bộ nhớ toàn cục
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // Đồng bộ hóa trong một threadBlock
        __syncthreads();
    }

    // Ghi kết quả cho block này vào bộ nhớ toàn cục
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8 (double *g_idata, double *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    double *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        double a1 = g_idata[idx];
        double a2 = g_idata[idx + blockDim.x];
        double a3 = g_idata[idx + 2 * blockDim.x];
        double a4 = g_idata[idx + 3 * blockDim.x];
        double b1 = g_idata[idx + 4 * blockDim.x];
        double b2 = g_idata[idx + 5 * blockDim.x];
        double b3 = g_idata[idx + 6 * blockDim.x];
        double b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads(); // Đồng bộ hóa các group data trong 8 thread kết cận

    // Thực hiện tính tổng ở bộ nhớ toàn cục
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }

        // Đồng bộ hóa trong một threadBlock
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8 (double *g_idata, double *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    double *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n)
    {
        double a1 = g_idata[idx];
        double a2 = g_idata[idx + blockDim.x];
        double a3 = g_idata[idx + 2 * blockDim.x];
        double a4 = g_idata[idx + 3 * blockDim.x];
        double b1 = g_idata[idx + 4 * blockDim.x];
        double b2 = g_idata[idx + 5 * blockDim.x];
        double b3 = g_idata[idx + 6 * blockDim.x];
        double b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    // Thực hiện tính tổng ở bộ nhớ toàn cục
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1){
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }

        // Đồng bộ hóa trong một threadBlock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32)
    {
        volatile double *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8 (double *g_idata, double *g_odata,
        unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    double *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n)
    {
        double a1 = g_idata[idx];
        double a2 = g_idata[idx + blockDim.x];
        double a3 = g_idata[idx + 2 * blockDim.x];
        double a4 = g_idata[idx + 3 * blockDim.x];
        double b1 = g_idata[idx + 4 * blockDim.x];
        double b2 = g_idata[idx + 5 * blockDim.x];
        double b3 = g_idata[idx + 6 * blockDim.x];
        double b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads(); // Đồng bộ hóa tất cả các thread trong một block

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile double *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // Ghi kết quả cho block này vào bộ nhớ toàn cục
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(double *g_idata, double *g_odata,
                                     unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // Chuyển từ con trỏ toàn cục sang con trỏ của block này
    double *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        double a1 = g_idata[idx];
        double a2 = g_idata[idx + blockDim.x];
        double a3 = g_idata[idx + 2 * blockDim.x];
        double a4 = g_idata[idx + 3 * blockDim.x];
        double b1 = g_idata[idx + 4 * blockDim.x];
        double b2 = g_idata[idx + 5 * blockDim.x];
        double b3 = g_idata[idx + 6 * blockDim.x];
        double b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads(); // Đồng bộ tất các thread trong một block

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile double *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // Ghi kết quả cho block này vào bộ nhớ toàn cục
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps (double *g_idata, double *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Chuyển từ con trỏ toàn cục sang con trỏ của block này
    double *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // Thực hiện tính tổng ở bộ nhớ toàn cục
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // Đồng bộ hóa trong một threadBlock
        __syncthreads();
    }

    // unrolling last warp
    if (tid < 32)
    {
        volatile double *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main()
{
    printf("############ THÔNG TIN GPU ############\n");
    // Chọn GPU thực thi câu lệnh    
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Khởi tạo kích thước vector
    int size = 1 << 24; // 2^24
    printf("Kích thước mảng : %d\n", size);
    
    // Kernel được cấu hình với 1D grid và 1D blocks
    int const BLOCK_SIZE = 512;
    dim3 block (BLOCK_SIZE, 1); // Block size có kích thước 512 x 1 ~ (x, y)
    dim3 grid  ((size + block.x - 1) / block.x, 1); // Grid size có kích thước ceil(size/block.x)
    printf("Kích thước : <<<Grid (%d, %d), Block (%d, %d)>>>\n", block.x, block.y, grid.x, grid.y);
 
    // Cấp phát bộ nhớ trên host (CPU)
    size_t bytes = size * sizeof(double);
    double *h_idata = (double *) malloc(bytes); // host input data
    double *h_odata = (double *) malloc(grid.x * sizeof(double)); // host output data
    double *temp    = (double *) malloc(bytes); // vùng nhớ tạp để copy input cho nhiều hàm thực thi khác nhau
 
    initialData(h_idata, size);
    // Copy vào biến temp để chạy với CPU
    memcpy (temp, h_idata, bytes);
 
    // Biến tính thời gian chạy
    double iStart, iElaps;
    double gpu_sum = 0.0; // hàm tính tổng kết quả trên GPU
    double gpu_bytes = grid.x * sizeof(double);

    // Cấp phát bộ nhớ trên device (GPU)
    double *d_idata = NULL;
    double *d_odata = NULL;
    CHECK(cudaMalloc(&d_idata, bytes));
    CHECK(cudaMalloc(&d_odata, gpu_bytes));
 
    printf("ID| Time \t\t| Sum result \t\t| <<<GridSize, BlockSize >>> | Kernel\t\t\n");
    // ############ 1. CPU #############
    iStart = seconds();
    double cpu_sum = recursiveReduce (temp, size);
    iElaps = seconds() - iStart;
    printf("1 | %f sec\t| %f\t|\t\t | recursiveReduce-CPU\n", iElaps, cpu_sum);

    // ############ 2. reduceNeighbored ############
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, gpu_bytes, cudaMemcpyDeviceToHost));
    gpu_sum = sumGPU(h_odata, grid.x);
    printf("2 | %f sec\t| %f\t|<<<%d, %d>>> | reduceNeighbored\n", iElaps, gpu_sum, grid.x, block.x);

    // ############ 3. reduceNeighboredLess ############
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, gpu_bytes, cudaMemcpyDeviceToHost));
    gpu_sum = sumGPU(h_odata, grid.x);
    printf("3 | %f sec\t| %f\t|<<<%d, %d>>> | reduceNeighboredLess\n", iElaps, gpu_sum, grid.x, block.x);

    // ############ 4. reduceInterleaved ############
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, gpu_bytes, cudaMemcpyDeviceToHost));
    gpu_sum = sumGPU(h_odata, grid.x);
    printf("4 | %f sec\t| %f\t|<<<%d, %d>>> | reduceInterleaved\n", iElaps, gpu_sum, grid.x, block.x);

    // ############ 5. reduceUnrolling2 ############
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(double), cudaMemcpyDeviceToHost));
    gpu_sum = sumGPU(h_odata, grid.x / 2);
    printf("5 | %f sec\t| %f\t|<<<%d, %d>>> | reduceUnrolling2\n", iElaps, gpu_sum, grid.x/2, block.x);

    // ############ 6. reduceUnrolling4 ############
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(double), cudaMemcpyDeviceToHost));
    gpu_sum = sumGPU(h_odata, grid.x / 4);
    printf("6 | %f sec\t| %f\t|<<<%d, %d>>> | reduceUnrolling4\n", iElaps, gpu_sum, grid.x/4, block.x);

    // ############ 7. reduceUnrolling8 ############
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(double), cudaMemcpyDeviceToHost));
    gpu_sum = sumGPU(h_odata, grid.x / 8);
    printf("7 | %f sec\t| %f\t|<<<%d, %d>>> | reduceUnrolling8\n", iElaps, gpu_sum, grid.x/8, block.x);

    // ############ 8. reduceUnrollWarps8 ############
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(double), cudaMemcpyDeviceToHost));
    gpu_sum = sumGPU(h_odata, grid.x / 8);
    printf("8 | %f sec\t| %f\t|<<<%d, %d>>> | reduceUnrollWarps8\n", iElaps, gpu_sum, grid.x/8, block.x);

    // ############ 9. reduceCompleteUnrollWarsp8 ############
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(double), cudaMemcpyDeviceToHost));
    gpu_sum = sumGPU(h_odata, grid.x / 8);
    printf("9 | %f sec\t| %f\t|<<<%d, %d>>> | reduceCompleteUnrollWarsp8\n", iElaps, gpu_sum, grid.x/8, block.x);

    // ############ 10. reduceCompleteUnroll ############
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    switch (BLOCK_SIZE){
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 512:
        reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 256:
        reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 128:
        reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 64:
        reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    }

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(double), cudaMemcpyDeviceToHost));
    gpu_sum = sumGPU(h_odata, grid.x / 8);
    printf("10| %f sec\t| %f\t|<<<%d, %d>>> | reduceCompleteUnroll\n", iElaps, gpu_sum, grid.x/8, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // Print sum result
    printf("Sum on CPU : %f\nSum on GPU : %f", cpu_sum, gpu_sum);

    return 0;
}