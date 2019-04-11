#include <QtWidgets>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

class Processor
{
public:
    static void InitCUDA(bool verbose);
    static void InitOpenCL(bool verbose);

    static QImage blurImageOpenMP(const QImage &image);
    static QImage blurImageCUDA(const QImage &image);
    static QImage blurImageOpenCL(const QImage &image);

    static QImage invertImageOpenMP(const QImage &image);
    static QImage invertImageCUDA(const QImage &image);
    static QImage invertImageOpenCL(const QImage &image);

private:
    static QImage processWithOpenMP(const QImage &image,
        void (*perPixelKernel)(const unsigned char originalImage[], unsigned char blurredImage[],
            unsigned int width, unsigned int height, unsigned int row, unsigned int col));
    static QImage processWithCUDA(const QImage &image,
        void (*kernel)(const unsigned char originalImage[], unsigned char blurredImage[], unsigned int width, unsigned int height));
    static QImage processWithOpenCL(const QImage &image, cl_kernel kernel, size_t maxWGS);

    static int cudaMaxThreadsPerBlock;

    static size_t clBlurKernelMaxWGS;
    static size_t clInvertKernelMaxWGS;
    static cl_program clProgram;
    static cl_kernel clBlurKernel;
    static cl_kernel clInvertKernel;
    static cl_context clContext;
    static cl_command_queue clQueue;
};