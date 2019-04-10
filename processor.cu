#include "processor.hpp"

#include <cuda.h>
#include <cmath>
#include <iostream>
#include <omp.h>


void Fatal(const char* format , ...)
{
    va_list args;
    va_start(args,format);
    vfprintf(stderr,format,args);
    va_end(args);
    exit(1);
}

int Processor::maxThreadsPerBlock;

__host__ __device__ unsigned char redComponent(const unsigned char image[], unsigned int idx)
{
    return image[idx*4+2];
}

__host__ __device__ unsigned char greenComponent(const unsigned char image[], unsigned int idx)
{
    return image[idx*4+1];
}

__host__ __device__ unsigned char blueComponent(const unsigned char image[], unsigned int idx)
{
    return image[idx*4];
}

__host__ __device__ void setUnusedComponent(unsigned char image[], unsigned int idx)
{
    // Unused byte should have value of 0xff
    image[idx*4+3] = 0xff;
}

__host__ __device__ void setRedComponent(unsigned char image[], unsigned int idx, unsigned char val)
{
    image[idx*4+2] = val;
}

__host__ __device__ void setGreenComponent(unsigned char image[], unsigned int idx, unsigned char val)
{
    image[idx*4+1] = val;
}


__host__ __device__ void setBlueComponent(unsigned char image[], unsigned int idx, unsigned char val)
{
    image[idx*4] = val;
}

__host__ __device__ void blurPixel(const unsigned char originalImage[], unsigned char blurredImage[],
    unsigned int width, unsigned int height, unsigned int row, unsigned int col)
{
    unsigned int idx = row*width + col;

    // Initialize component values to current value
    unsigned int contribs = 1;
    unsigned int red = redComponent(originalImage, idx);
    unsigned int blue = blueComponent(originalImage, idx);
    unsigned int green = greenComponent(originalImage, idx);

    // Determine if pixel is at any of the borders
    bool bordersLeft = col == 0;
    bool bordersRight = col == width-1;
    bool bordersTop = row == 0;
    bool bordersBottom = row == height-1;

    // Add all pixels
    // Top-left pixel
    if (!bordersLeft && !bordersTop) {
        contribs++;
        unsigned int pixelIdx = (row-1)*width + (col-1);
        red += redComponent(originalImage, pixelIdx);
        blue += blueComponent(originalImage, pixelIdx);
        green += greenComponent(originalImage, pixelIdx);
    }
    // Top pixel
    if (!bordersTop) {
        contribs++;
        unsigned int pixelIdx = (row-1)*width + (col);
        red += redComponent(originalImage, pixelIdx);
        blue += blueComponent(originalImage, pixelIdx);
        green += greenComponent(originalImage, pixelIdx);
    }
    // Top-right pixel
    if (!bordersRight && !bordersTop) {
        contribs++;
        unsigned int pixelIdx = (row-1)*width + (col+1);
        red += redComponent(originalImage, pixelIdx);
        blue += blueComponent(originalImage, pixelIdx);
        green += greenComponent(originalImage, pixelIdx);
    }
    // Left pixel
    if (!bordersLeft) {
        contribs++;
        unsigned int pixelIdx = (row)*width + (col-1);
        red += redComponent(originalImage, pixelIdx);
        blue += blueComponent(originalImage, pixelIdx);
        green += greenComponent(originalImage, pixelIdx);
    }
    // Right pixel
    if (!bordersRight) {
        contribs++;
        unsigned int pixelIdx = (row)*width + (col+1);
        red += redComponent(originalImage, pixelIdx);
        blue += blueComponent(originalImage, pixelIdx);
        green += greenComponent(originalImage, pixelIdx);
    }
    // Bottom-left pixel
    if (!bordersLeft && !bordersBottom) {
        contribs++;
        unsigned int pixelIdx = (row+1)*width + (col-1);
        red += redComponent(originalImage, pixelIdx);
        blue += blueComponent(originalImage, pixelIdx);
        green += greenComponent(originalImage, pixelIdx);
    }
    // Bottom pixel
    if (!bordersBottom) {
        contribs++;
        unsigned int pixelIdx = (row+1)*width + (col);
        red += redComponent(originalImage, pixelIdx);
        blue += blueComponent(originalImage, pixelIdx);
        green += greenComponent(originalImage, pixelIdx);
    }
    // Bottom-right pixel
    if (!bordersRight && !bordersBottom) {
        contribs++;
        unsigned int pixelIdx = (row+1)*width + (col+1);
        red += redComponent(originalImage, pixelIdx);
        blue += blueComponent(originalImage, pixelIdx);
        green += greenComponent(originalImage, pixelIdx);
    }

    // Average results and set modified pixel
    setUnusedComponent(blurredImage, idx);
    setRedComponent(blurredImage, idx, (unsigned char) (red/contribs));
    setGreenComponent(blurredImage, idx, (unsigned char) (green/contribs));
    setBlueComponent(blurredImage, idx, (unsigned char) (blue/contribs));
}

__global__ void blurKernel(const unsigned char originalImage[], unsigned char blurredImage[], unsigned int width, unsigned int height)
{
    // Determine current pixel
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = idx / width;
    unsigned int col = idx % width;

    // Do nothing if past end
    if (idx >= width*height) return;

    blurPixel(originalImage, blurredImage, width, height, row, col);
}

__host__ __device__ void invertPixel(const unsigned char originalImage[], unsigned char invertedImage[],
    unsigned int width, unsigned int height, unsigned int row, unsigned int col)
{
    unsigned int idx = row*width + col;

    // Fetch current values
    unsigned char red = redComponent(originalImage, idx);
    unsigned char blue = blueComponent(originalImage, idx);
    unsigned char green = greenComponent(originalImage, idx);

    // Set inverted values
    setUnusedComponent(invertedImage, idx);
    setRedComponent(invertedImage, idx, 255-red);
    setBlueComponent(invertedImage, idx, 255-blue);
    setGreenComponent(invertedImage, idx, 255-green);
}

__global__ void invertKernel(const unsigned char originalImage[], unsigned char invertedImage[], unsigned int width, unsigned int height)
{
    // Determine current pixel
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = idx / width;
    unsigned int col = idx % width;

    // Do nothing if past end
    if (idx >= width*height) return;

    invertPixel(originalImage, invertedImage, width, height, row, col);
}

QImage Processor::blurImage(const QImage &image)
{
    unsigned long numBytes = image.bytesPerLine()*image.height();

    // Allocate memory for image on GPU
    unsigned char *initialImageD;
    unsigned char *finalImageD;
    if (cudaMalloc((void**)&initialImageD, numBytes))
        Fatal("Cannot allocate space for initial image on device.\n");
    if (cudaMalloc((void**)&finalImageD, numBytes))
        Fatal("Cannot allocate space for modified image on device.\n");

    // Copy initial image from host to device
    if (cudaMemcpy(initialImageD, image.bits(), numBytes, cudaMemcpyHostToDevice))
        Fatal("Cannot transfer initial image from host to device.\n");

    // Execute kernel
    dim3 blockSize(maxThreadsPerBlock, 1, 1);
    dim3 gridSize(ceil(image.width()*image.height()/((float)maxThreadsPerBlock)), 1, 1);
    blurKernel<<<gridSize, blockSize>>>(initialImageD, finalImageD, image.width(), image.height());
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err) Fatal("blurKernel failure: %s\n", cudaGetErrorString(err));

    // Copy modified image from device to host
    unsigned char *finalImageH = (unsigned char*)aligned_alloc(32, numBytes);
    if (cudaMemcpy(finalImageH, finalImageD, numBytes, cudaMemcpyDeviceToHost))
        Fatal("Cannot transfer modified image from device to host.\n");

    // Free device memory
    cudaFree(initialImageD);
    cudaFree(finalImageD);


    // Construct image from binary data
    QImage finalImage(finalImageH, image.width(), image.height(), image.format());
    return finalImage;
}

QImage Processor::blurImageCPU(const QImage &image)
{
    // Allocate memory for new image
    unsigned long numBytes = image.bytesPerLine()*image.height();
    unsigned char *finalImageBin = (unsigned char*)aligned_alloc(32, numBytes);

    // Calculate the modified version of each pixel
    const uchar *imageBits = image.bits();
    for (int i = 0; i < image.height(); i++) {
        for (int j = 0; j < image.width(); j++) {
            blurPixel(imageBits, finalImageBin, image.width(), image.height(), i, j);
        }
    }

    // Construct image from binary data
    QImage finalImage(finalImageBin, image.width(), image.height(), image.format());
    return finalImage;
}

QImage Processor::invertImage(const QImage &image)
{
    unsigned long numBytes = image.bytesPerLine()*image.height();

    // Allocate memory for image on GPU
    unsigned char *initialImageD;
    unsigned char *finalImageD;
    if (cudaMalloc((void**)&initialImageD, numBytes))
        Fatal("Cannot allocate space for initial image on device.\n");
    if (cudaMalloc((void**)&finalImageD, numBytes))
        Fatal("Cannot allocate space for modified image on device.\n");

    // Copy initial image from host to device
    if (cudaMemcpy(initialImageD, image.bits(), numBytes, cudaMemcpyHostToDevice))
        Fatal("Cannot transfer initial image from host to device.\n");

    // Execute kernel
    dim3 blockSize(maxThreadsPerBlock, 1, 1);
    dim3 gridSize(ceil(image.width()*image.height()/((float)maxThreadsPerBlock)), 1, 1);
    invertKernel<<<gridSize, blockSize>>>(initialImageD, finalImageD, image.width(), image.height());
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err) Fatal("invertKernel failure: %s\n", cudaGetErrorString(err));

    // Copy modified image from device to host
    unsigned char *finalImageH = (unsigned char*)aligned_alloc(32, numBytes);
    if (cudaMemcpy(finalImageH, finalImageD, numBytes, cudaMemcpyDeviceToHost))
        Fatal("Cannot transfer modified image from device to host.\n");

    // Free device memory
    cudaFree(initialImageD);
    cudaFree(finalImageD);


    // Construct image from binary data
    QImage finalImage(finalImageH, image.width(), image.height(), image.format());
    return finalImage;
}

QImage Processor::invertImageCPU(const QImage &image)
{
    // Allocate memory for new image
    unsigned long numBytes = image.bytesPerLine()*image.height();
    unsigned char *finalImageBin = (unsigned char*)aligned_alloc(32, numBytes);

    // Calculate the modified version of each pixel
    const uchar *imageBits = image.bits();
    #pragma omp parallel for
    for (int i = 0; i < image.height(); i++) {
        for (int j = 0; j < image.width(); j++) {
            invertPixel(imageBits, finalImageBin, image.width(), image.height(), i, j);
        }
    }

    // Construct image from binary data
    QImage finalImage(finalImageBin, image.width(), image.height(), image.format());
    return finalImage;
}


void Processor::InitGPU(bool verbose)
{
    //  Get number of CUDA devices
    int num;
    if (cudaGetDeviceCount(&num)) Fatal("Cannot get number of CUDA devices\n");
    if (num<1) Fatal("No CUDA devices found\n");

    //  Get fastest device
    cudaDeviceProp prop;
    int   MaxDevice = -1;
    int   MaxGflops = -1;
    for (int dev=0;dev<num;dev++)
    {
        if (cudaGetDeviceProperties(&prop,dev)) Fatal("Error getting device %d properties\n",dev);
        int Gflops = prop.multiProcessorCount * prop.clockRate;
        if (verbose)
            printf("CUDA Device %d: %s Gflops %f Processors %d Threads/Block %d Shared Mem %lu\n",
                dev, prop.name, 1e-6*Gflops, prop.multiProcessorCount, prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
        if(Gflops > MaxGflops)
        {
            MaxGflops = Gflops;
            MaxDevice = dev;
        }
    }

    //  Print and set device
    if (cudaGetDeviceProperties(&prop,MaxDevice)) Fatal("Error getting device %d properties\n",MaxDevice);
    printf("Fastest CUDA Device %d: %s\n",MaxDevice,prop.name);
    cudaSetDevice(MaxDevice);

    //  Save max thread count
    maxThreadsPerBlock = prop.maxThreadsPerBlock;
}