/*
Tham khảo nguồn : https://gist.github.com/stevendborrelli/4286842
*/

#include <cuda.h>
#include <stdio.h>

void CHECK(int error, char *message, char *file, int line)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Lỗi CUDA: %s : %i. Ở %s dòng %d\n", message, error, file, line);
        exit(-1);
    }
}

int main(int argc, char **argv)
{
    int deviceCount;
    CHECK(cudaGetDeviceCount(&deviceCount), "GetDeviceCount", __FILE__, __LINE__);
    printf("Số lượng CUDA device : %d.\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++){
        printf("****************** DEVICE %d ******************\n", (dev+1));
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev), "Thông tin device", __FILE__, __LINE__);

        if (dev == 0){
            if (deviceProp.major == 9999 && deviceProp.minor == 9999){
                printf("Không tìm thấy CUDA device nào !\n");
                return -1;
            }
            else if (deviceCount == 1){
                printf("Có một device hỗ trợ CUDA\n");
            }
            else{
                printf("Có %d hỗ trợ CUDA\n", deviceCount);
            }
        }

        printf("Tên Device: %s\n", deviceProp.name);
        printf("Số revision nhiều: %d\n", deviceProp.major);
        printf("Số revision nhỏ: %d\n", deviceProp.minor);
        printf("Tổng kích thước bộ nhớ toàn cục : %d\n", deviceProp.totalGlobalMem);
        printf("Tổng kích thước bộ nhớ chia sẻ trên một block : %d\n", deviceProp.sharedMemPerBlock);
        printf("Tổng kích thước bộ nhớ hằng : %d\n", deviceProp.totalConstMem);
        printf("Kích thước Warp: %d\n", deviceProp.warpSize);
        printf("Kích thước block tối đa: %d x %d x %d\n", deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);

        printf("Kích thước grid tối đa: %d x %d x %d\n", deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("Tỉ lệ đồng hồ: %d\n", deviceProp.clockRate);
        printf("Số lượng đa xử lý: %d\n", deviceProp.multiProcessorCount);
    }

    return 0;
}