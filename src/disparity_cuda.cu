
#include <stdio.h>
#include <assert.h>

#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include <CImg.h>

using namespace cimg_library;

bool hasCudaDevice()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        return false;
    }

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
        return false;
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device: %s\n", deviceProp.name);

    int driverVersion, runtimeVersion = 0;

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    return true;
}

int main(int argc, char** argv)
{
    if (!hasCudaDevice())
        exit(EXIT_FAILURE);

    CImg<unsigned char> image("samples/kitti_stereo_2012/000000_10_L.png");

    exit(EXIT_SUCCESS);
}