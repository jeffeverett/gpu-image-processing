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
    static int maxThreadsPerBlock;
};