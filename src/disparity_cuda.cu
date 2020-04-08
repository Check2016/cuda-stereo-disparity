
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include <helper_cuda.h>

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

CImg<unsigned char> createGrayscale(CImg<unsigned char> img)
{
    CImg<unsigned char> gray(img.width(), img.height(), 1, 1);

    cimg_forXY(img,x,y) gray(x,y) = 0.2126f * img(x,y,0,0) + 0.7152f * img(x,y,0,1) + 0.0722f * img(x,y,0,2);

    return gray;
}

__global__ void CensusTransform(unsigned char* img, unsigned int* census, int census_w)
{
    int img_w = census_w + 4;

    unsigned int census_i = blockIdx.x + blockIdx.y * census_w;
    unsigned int img_center_i = (blockIdx.x+2) + (blockIdx.y+2) * img_w;
    unsigned int img_top_left_i = blockIdx.x + blockIdx.y * img_w;

    for(uint8_t y = 0; y < 5; y++)
    {
        for(uint8_t x = 0; x < 5; x++)
        {
            if (y == x == 2) continue;

            unsigned int pos = img_top_left_i + x + y * img_w;

            census[census_i] = (census[census_i] << 1) | (img[img_center_i] <= img[pos]);
        }
    }
}

int main(int argc, char** argv)
{
    if (!hasCudaDevice())
        exit(EXIT_FAILURE);

    CImg<unsigned char> image("samples/kitti_stereo_2012/000000_10_L.png");
    unsigned int width = image.width();
    unsigned int height = image.height();

    CImg<unsigned char> image_gray = createGrayscale(image);
    unsigned int image_gray_bytes = width*height*sizeof(unsigned char);

    unsigned int census_width = width - 4;
    unsigned int census_height = height - 4;
    CImg<unsigned int> census(census_width, census_height, 1, 1);
    unsigned int census_bytes = census_width*census_height*sizeof(unsigned int);

    unsigned char* h_image_gray = image_gray.data();
    unsigned int* h_census = census.data();

    unsigned char* d_image_gray;
    unsigned int* d_census;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_image_gray), image_gray_bytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_census), census_bytes));

    checkCudaErrors(cudaMemcpy(d_image_gray, h_image_gray, image_gray_bytes, cudaMemcpyHostToDevice));

    dim3 block(1, 1);
    dim3 grid(width - 4, height - 4);

    CensusTransform<<<grid, block>>>(d_image_gray, d_census, census_width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    checkCudaErrors(cudaMemcpy(h_census, d_census, census_bytes, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_image_gray));
    checkCudaErrors(cudaFree(d_census));

    CImgDisplay image_disp(image_gray, "Image");
    CImgDisplay census_disp(census, "Census", 1);
    while(!image_disp.is_closed() && !census_disp.is_closed())
    {
        image_disp.wait();
    }

    exit(EXIT_SUCCESS);
}