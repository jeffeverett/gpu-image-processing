#include <QtWidgets>

class Processor
{
public:
    static void InitGPU(bool verbose);

    static QImage blurImage(const QImage &image);
    static QImage blurImageCPU(const QImage &image);

    static QImage invertImage(const QImage &image);
    static QImage invertImageCPU(const QImage &image);

private:
    static QImage processWithCUDA(const QImage &image,
        void (*kernel)(const unsigned char originalImage[], unsigned char blurredImage[], unsigned int width, unsigned int height));

    static int maxThreadsPerBlock;
};