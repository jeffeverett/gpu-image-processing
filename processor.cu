#include "processor.hpp"

#include <cuda.h>
#include <cmath>
#include <iostream>
#include <omp.h>


/**************** Utility functions ****************/
void Fatal(const char* format , ...)
{
    va_list args;
    va_start(args,format);
    vfprintf(stderr,format,args);
    va_end(args);
    exit(1);
}

void Notify(const char* errinfo,const void* private_info,size_t cb,void* user_data)
{
    fprintf(stderr,"%s\n",errinfo);
}

/**************** CODE FOR OpenMP/CUDA ****************/
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

/**************** CODE FOR OpenCL ****************/
const char *source =
    "unsigned char redComponent(const unsigned char image[], unsigned int idx)\n"
    "{\n"
    "    return image[idx*4+2];\n"
    "}"

    "unsigned char greenComponent(const unsigned char image[], unsigned int idx)\n"
    "{\n"
    "    return image[idx*4+1];\n"
    "}\n"

    "unsigned char blueComponent(const unsigned char image[], unsigned int idx)\n"
    "{\n"
    "    return image[idx*4];\n"
    "}\n"

    "void setUnusedComponent(unsigned char image[], unsigned int idx)\n"
    "{\n"
    "    // Unused byte should have value of 0xff\n"
    "    image[idx*4+3] = 0xff;\n"
    "}\n"

    "void setRedComponent(unsigned char image[], unsigned int idx, unsigned char val)\n"
    "{\n"
    "    image[idx*4+2] = val;\n"
    "}\n"

    "void setGreenComponent(unsigned char image[], unsigned int idx, unsigned char val)\n"
    "{\n"
    "    image[idx*4+1] = val;\n"
    "}\n"

    "void setBlueComponent(unsigned char image[], unsigned int idx, unsigned char val)\n"
    "{\n"
    "    image[idx*4] = val;\n"
    "}\n"

    "void blurPixel(const unsigned char originalImage[], unsigned char blurredImage[],\n"
    "    unsigned int width, unsigned int height, unsigned int row, unsigned int col)\n"
    "{\n"
    "    unsigned int idx = row*width + col;\n"
    "\n"
    "    // Initialize component values to current value\n"
    "    unsigned int contribs = 1;\n"
    "    unsigned int red = redComponent(originalImage, idx);\n"
    "    unsigned int blue = blueComponent(originalImage, idx);\n"
    "    unsigned int green = greenComponent(originalImage, idx);\n"
    "\n"
    "    // Determine if pixel is at any of the borders\n"
    "    bool bordersLeft = col == 0;\n"
    "    bool bordersRight = col == width-1;\n"
    "    bool bordersTop = row == 0;\n"
    "    bool bordersBottom = row == height-1;\n"
    "\n"
    "    // Add all pixels\n"
    "    // Top-left pixel\n"
    "    if (!bordersLeft && !bordersTop) {\n"
    "        contribs++;\n"
    "        unsigned int pixelIdx = (row-1)*width + (col-1);\n"
    "        red += redComponent(originalImage, pixelIdx);\n"
    "        blue += blueComponent(originalImage, pixelIdx);\n"
    "        green += greenComponent(originalImage, pixelIdx);\n"
    "    }\n"
    "    // Top pixel\n"
    "    if (!bordersTop) {\n"
    "        contribs++;\n"
    "        unsigned int pixelIdx = (row-1)*width + (col);\n"
    "        red += redComponent(originalImage, pixelIdx);\n"
    "        blue += blueComponent(originalImage, pixelIdx);\n"
    "        green += greenComponent(originalImage, pixelIdx);\n"
    "    }\n"
    "    // Top-right pixel\n"
    "    if (!bordersRight && !bordersTop) {\n"
    "        contribs++;\n"
    "        unsigned int pixelIdx = (row-1)*width + (col+1);\n"
    "        red += redComponent(originalImage, pixelIdx);\n"
    "        blue += blueComponent(originalImage, pixelIdx);\n"
    "        green += greenComponent(originalImage, pixelIdx);\n"
    "    }\n"
    "    // Left pixel\n"
    "    if (!bordersLeft) {\n"
    "        contribs++;\n"
    "        unsigned int pixelIdx = (row)*width + (col-1);\n"
    "        red += redComponent(originalImage, pixelIdx);\n"
    "        blue += blueComponent(originalImage, pixelIdx);\n"
    "        green += greenComponent(originalImage, pixelIdx);\n"
    "    }\n"
    "    // Right pixel\n"
    "    if (!bordersRight) {\n"
    "        contribs++;\n"
    "        unsigned int pixelIdx = (row)*width + (col+1);\n"
    "        red += redComponent(originalImage, pixelIdx);\n"
    "        blue += blueComponent(originalImage, pixelIdx);\n"
    "        green += greenComponent(originalImage, pixelIdx);\n"
    "    }\n"
    "    // Bottom-left pixel\n"
    "    if (!bordersLeft && !bordersBottom) {\n"
    "        contribs++;\n"
    "        unsigned int pixelIdx = (row+1)*width + (col-1);\n"
    "        red += redComponent(originalImage, pixelIdx);\n"
    "        blue += blueComponent(originalImage, pixelIdx);\n"
    "        green += greenComponent(originalImage, pixelIdx);\n"
    "    }\n"
    "    // Bottom pixel\n"
    "    if (!bordersBottom) {\n"
    "        contribs++;\n"
    "        unsigned int pixelIdx = (row+1)*width + (col);\n"
    "        red += redComponent(originalImage, pixelIdx);\n"
    "        blue += blueComponent(originalImage, pixelIdx);\n"
    "        green += greenComponent(originalImage, pixelIdx);\n"
    "    }\n"
    "    // Bottom-right pixel\n"
    "    if (!bordersRight && !bordersBottom) {\n"
    "        contribs++;\n"
    "        unsigned int pixelIdx = (row+1)*width + (col+1);\n"
    "        red += redComponent(originalImage, pixelIdx);\n"
    "        blue += blueComponent(originalImage, pixelIdx);\n"
    "        green += greenComponent(originalImage, pixelIdx);\n"
    "    }\n"
    "\n"
    "    // Average results and set modified pixel\n"
    "    setUnusedComponent(blurredImage, idx);\n"
    "    setRedComponent(blurredImage, idx, (unsigned char) (red/contribs));\n"
    "    setGreenComponent(blurredImage, idx, (unsigned char) (green/contribs));\n"
    "    setBlueComponent(blurredImage, idx, (unsigned char) (blue/contribs));\n"
    "}\n"

    "__kernel void blurKernel(__global const unsigned char originalImage[], __global const unsigned char blurredImage[],"
    "   const unsigned int width, const unsigned int height)\n"
    "{\n"
    "   // Determine current pixel\n"
    "   unsigned int idx = get_global_id(0);\n"
    "   unsigned int row = idx / width;\n"
    "   unsigned int col = idx % width;\n"
    "\n"
    "   // Do nothing if past end\n"
    "   if (idx >= width*height) return;\n"
    "\n"
    "   blurPixel(originalImage, blurredImage, width, height, row, col);\n"
    "}\n"

    "void invertPixel(const unsigned char originalImage[], unsigned char invertedImage[],\n"
    "    unsigned int width, unsigned int height, unsigned int row, unsigned int col)\n"
    "{\n"
    "    unsigned int idx = row*width + col;\n"
    "\n"
    "    // Fetch current values\n"
    "    unsigned char red = redComponent(originalImage, idx);\n"
    "    unsigned char blue = blueComponent(originalImage, idx);\n"
    "    unsigned char green = greenComponent(originalImage, idx);\n"
    "\n"
    "    // Set inverted values\n"
    "    setUnusedComponent(invertedImage, idx);\n"
    "    setRedComponent(invertedImage, idx, 255-red);\n"
    "    setBlueComponent(invertedImage, idx, 255-blue);\n"
    "    setGreenComponent(invertedImage, idx, 255-green);\n"
    "}\n"

    "__kernel void invertKernel(__global const unsigned char originalImage[], __global const unsigned char blurredImage[],"
    "   const unsigned int width, const unsigned int height)\n"
    "{\n"
    "   // Determine current pixel\n"
    "   unsigned int idx = get_global_id(0);\n"
    "   unsigned int row = idx / width;\n"
    "   unsigned int col = idx % width;\n"
    "\n"
    "   // Do nothing if past end\n"
    "   if (idx >= width*height) return;\n"
    "\n"
    "   invertPixel(originalImage, blurredImage, width, height, row, col);\n"
    "}\n";


/**************** CODE FOR Class Implementation ****************/
int Processor::cudaMaxThreadsPerBlock;

size_t Processor::clBlurKernelMaxWGS;
size_t Processor::clInvertKernelMaxWGS;
cl_program Processor::clProgram;
cl_kernel Processor::clBlurKernel;
cl_kernel Processor::clInvertKernel;
cl_context Processor::clContext;
cl_command_queue Processor::clQueue;


// Image blur methods
QImage Processor::blurImageOpenMP(const QImage &image)
{
    return processWithOpenMP(image, blurPixel);
}

QImage Processor::blurImageCUDA(const QImage &image)
{
    return processWithCUDA(image, blurKernel);
}

QImage Processor::blurImageOpenCL(const QImage &image)
{
    return processWithOpenCL(image, clBlurKernel, clBlurKernelMaxWGS);
}


// Image inversion methods
QImage Processor::invertImageOpenMP(const QImage &image)
{
    return processWithOpenMP(image, invertPixel);
}

QImage Processor::invertImageCUDA(const QImage &image)
{
    return processWithCUDA(image, invertKernel);
}

QImage Processor::invertImageOpenCL(const QImage &image)
{
    return processWithOpenCL(image, clInvertKernel, clInvertKernelMaxWGS);
}

// Processing methods
QImage Processor::processWithOpenMP(const QImage &image,
    void (*perPixelKernel)(const unsigned char originalImage[], unsigned char blurredImage[],
        unsigned int width, unsigned int height, unsigned int row, unsigned int col))
{
    // Allocate memory for new image
    unsigned long numBytes = image.bytesPerLine()*image.height();
    unsigned char *finalImageBin = (unsigned char*)aligned_alloc(32, numBytes);

    // Calculate the modified version of each pixel
    const uchar *imageBits = image.bits();
    #pragma omp parallel for
    for (int i = 0; i < image.height(); i++) {
        for (int j = 0; j < image.width(); j++) {
            perPixelKernel(imageBits, finalImageBin, image.width(), image.height(), i, j);
        }
    }

    // Construct image from binary data
    QImage finalImage(finalImageBin, image.width(), image.height(), image.format());
    return finalImage;
}

QImage Processor::processWithCUDA(const QImage &image,
    void (*kernel)(const unsigned char originalImage[], unsigned char blurredImage[], unsigned int width, unsigned int height))
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
    dim3 blockSize(cudaMaxThreadsPerBlock, 1, 1);
    dim3 gridSize((unsigned int) ceil(image.width()*image.height()/((float)cudaMaxThreadsPerBlock)), 1, 1);
    kernel<<<gridSize, blockSize>>>(initialImageD, finalImageD, image.width(), image.height());
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err) Fatal("kernel failure: %s\n", cudaGetErrorString(err));

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

QImage Processor::processWithOpenCL(const QImage &image, cl_kernel kernel, size_t maxWGS)
{
    cl_int err;
    unsigned long numBytes = image.bytesPerLine()*image.height();

    // Create buffer with initial image contents on GPU
    cl_mem initialImageD = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBytes, (void*)image.bits(), &err);
    if (err)
        Fatal("Cannot create buffer for initial image on device.\n");

    // Allocate GPU memory for modified image
    cl_mem finalImageD = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, numBytes, nullptr, &err);
    if (err)
        Fatal("Cannot allocate space for modified image on device.\n");

    // Set parameters for kernel
    unsigned int imageWidth = image.width();
    unsigned int imageHeight = image.height();
    if (clSetKernelArg(kernel, 0, sizeof(cl_mem), &initialImageD)) Fatal("Cannot set kernel parameter initialImage\n");
    if (clSetKernelArg(kernel, 1, sizeof(cl_mem), &finalImageD)) Fatal("Cannot set kernel parameter finalImage\n");
    if (clSetKernelArg(kernel, 2, sizeof(unsigned int), &imageWidth)) Fatal("Cannot set kernel parameter width\n");
    if (clSetKernelArg(kernel, 3, sizeof(unsigned int), &imageHeight)) Fatal("Cannot set kernel parameter height\n");

    // Run kernel
    size_t localSize[1]  = {maxWGS};
    size_t globalSize[1] = {((size_t) ceil(image.width()*image.height()/(float) maxWGS))*maxWGS};
    err = clEnqueueNDRangeKernel(clQueue, kernel, 1, nullptr, globalSize, localSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        Fatal("Cannot run kernel: %d\n", err);

    // Copy modified image from device to host
    unsigned char *finalImageH = (unsigned char*)aligned_alloc(32, numBytes);
    if (clEnqueueReadBuffer(clQueue, finalImageD, CL_TRUE, 0, numBytes, finalImageH, 0, nullptr, nullptr))
        Fatal("Cannot transfer modified image from device to host.\n");

    // Free device memory
    clReleaseMemObject(initialImageD);
    clReleaseMemObject(finalImageD);


    // Construct image from binary data
    QImage finalImage(finalImageH, image.width(), image.height(), image.format());
    return finalImage;
}

// Setup methods
void Processor::InitCUDA(bool verbose)
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
    cudaMaxThreadsPerBlock = prop.maxThreadsPerBlock;
}

void Processor::InitOpenCL(bool verbose)
{
    cl_device_id devid;
    cl_uint Nplat;
    cl_int  err;
    char name[1024];
    int  MaxGflops = -1;

    //  Get platforms
    cl_platform_id platforms[1024];
    if (clGetPlatformIDs(1024,platforms,&Nplat))
        Fatal("Cannot get number of OpenCL platforms\n");
    else if (Nplat<1)
        Fatal("No OpenCL platforms found\n");
    //  Loop over platforms
    for (unsigned int platform=0;platform<Nplat;platform++)
    {
        if (clGetPlatformInfo(platforms[platform],CL_PLATFORM_NAME,sizeof(name),name,NULL)) Fatal("Cannot get OpenCL platform name\n");
        if (verbose) printf("OpenCL Platform %d: %s\n",platform,name);

        //  Get GPU device IDs
        cl_uint Ndev;
        cl_device_id id[1024];
        if (clGetDeviceIDs(platforms[platform],CL_DEVICE_TYPE_GPU,1024,id,&Ndev)) {
            continue;
        }
        else if (Ndev<1)
            Fatal("No OpenCL devices found\n");

        //  Find the fastest device
        for (unsigned int dev=0;dev<Ndev;dev++)
        {
            cl_uint proc,freq;
            if (clGetDeviceInfo(id[dev],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(proc),&proc,NULL)) Fatal("Cannot get OpenCL device units\n");
            if (clGetDeviceInfo(id[dev],CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(freq),&freq,NULL)) Fatal("Cannot get OpenCL device frequency\n");
            if (clGetDeviceInfo(id[dev],CL_DEVICE_NAME,sizeof(name),name, NULL)) Fatal("Cannot get OpenCL device name\n");
            int Gflops = proc*freq;
            if (verbose) printf("OpenCL Device %d: %s Gflops %f\n",dev,name,1e-3*Gflops);
            if(Gflops > MaxGflops)
            {
                devid = id[dev];
                MaxGflops = Gflops;
            }
        }
    }

    //  Print fastest device info
    if (clGetDeviceInfo(devid,CL_DEVICE_NAME,sizeof(name),name,NULL)) Fatal("Cannot get OpenCL device name\n");
    printf("Fastest OpenCL Device: %s\n",name);

    //  Create OpenCL context for fastest device
    clContext = clCreateContext(0,1,&devid,Notify,NULL,&err);
    if(!clContext || err) Fatal("Cannot create OpenCL context\n");

    //  Create OpenCL command queue for fastest device
    clQueue = clCreateCommandQueueWithProperties(clContext, devid, nullptr, &err);
    if(!clQueue || err) Fatal("Cannot create OpenCL command queue\n");


    //  Compile kernel
    clProgram = clCreateProgramWithSource(clContext,1,&source,0,&err);
    if (err) Fatal("Cannot create program\n");
    if (clBuildProgram(clProgram,0,NULL,NULL,NULL,NULL)) {
        char log[1048576];
        if (clGetProgramBuildInfo(clProgram,devid,CL_PROGRAM_BUILD_LOG,sizeof(log),log,NULL))
            Fatal("Cannot get build log\n");
        else
            Fatal("Cannot build program\n%s\n",log);
    }
    clBlurKernel = clCreateKernel(clProgram,"blurKernel",&err);
    if (err) Fatal("Cannot create blur kernel\n");
    clInvertKernel = clCreateKernel(clProgram,"invertKernel",&err);
    if (err) Fatal("Cannot create invert kernel\n");

    // Get per-kernel max work-group size
    if (clGetKernelWorkGroupInfo(clBlurKernel, devid, CL_KERNEL_WORK_GROUP_SIZE, sizeof(clBlurKernelMaxWGS), &clBlurKernelMaxWGS, nullptr))
        Fatal("Cannot get OpenCL max work group size for blur kernel\n");
    if (clGetKernelWorkGroupInfo(clInvertKernel, devid, CL_KERNEL_WORK_GROUP_SIZE, sizeof(clInvertKernelMaxWGS), &clInvertKernelMaxWGS, nullptr))
        Fatal("Cannot get OpenCL max work group size for invert kernel\n");
}