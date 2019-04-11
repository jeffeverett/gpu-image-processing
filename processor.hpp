#include <QtWidgets>

class Processor
{
public:
    static void InitGPU(bool verbose);

    static QImage blurImageOpenMP(const QImage &image);
    static QImage blurImageCUDA(const QImage &image);

    static QImage invertImageOpenMP(const QImage &image);
    static QImage invertImageCUDA(const QImage &image);

private:
    static QImage processWithOpenMP(const QImage &image,
        void (*perPixelKernel)(const unsigned char originalImage[], unsigned char blurredImage[],
            unsigned int width, unsigned int height, unsigned int row, unsigned int col));
    static QImage processWithCUDA(const QImage &image,
        void (*kernel)(const unsigned char originalImage[], unsigned char blurredImage[], unsigned int width, unsigned int height));

    static int maxThreadsPerBlock;
};