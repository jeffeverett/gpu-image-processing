#include "imageviewer.hpp"
#include "processor.hpp"

#include <QtWidgets>
#include <QGridLayout>

ImageViewer::ImageViewer()
   : centralWidget(new QWidget)
   , gridLayout(new QGridLayout)
   , initialImageLabel(new QLabel)
   , initialScrollArea(new QScrollArea)
   , scaleFactor(1)
   , finalImageLabel(new QLabel)
   , finalScrollArea(new QScrollArea)
   , modificationComboBox(new QComboBox)
   , modifyButton(new QPushButton("Modify"))
   , timingInfoLabel(new QLabel)
{
    initialImageLabel->setBackgroundRole(QPalette::Base);
    initialImageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    initialImageLabel->setScaledContents(true);

    initialScrollArea->setBackgroundRole(QPalette::Dark);
    initialScrollArea->setWidget(initialImageLabel);
    initialScrollArea->setVisible(false);

    finalImageLabel->setBackgroundRole(QPalette::Base);
    finalImageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    finalImageLabel->setScaledContents(true);

    finalScrollArea->setBackgroundRole(QPalette::Dark);
    finalScrollArea->setWidget(finalImageLabel);
    finalScrollArea->setVisible(false);

    modificationComboBox->addItem("Blur");
    modificationComboBox->addItem("Invert");
    modificationComboBox->setCurrentIndex(0);

    gridLayout->addWidget(initialScrollArea, 0, 0);
    gridLayout->addWidget(finalScrollArea, 0, 1);
    gridLayout->addWidget(modificationComboBox, 1, 0, 1, 2, Qt::AlignCenter);
    gridLayout->addWidget(modifyButton, 2, 0, 1, 2, Qt::AlignCenter);
    gridLayout->addWidget(timingInfoLabel, 3, 0, 1, 2);

    gridLayout->setColumnStretch(0,100);
    gridLayout->setColumnStretch(1,100);
    gridLayout->setRowStretch(0,100);

    centralWidget->setLayout(gridLayout);
    setCentralWidget(centralWidget);

    createActions();

    resize(QGuiApplication::primaryScreen()->availableSize() * 3 / 5);

    Processor::InitGPU(true);
}


bool ImageViewer::loadFile(const QString &fileName)
{
    QImageReader reader(fileName);
    reader.setAutoTransform(true);
    QImage newImage = reader.read();
    newImage = newImage.convertToFormat(QImage::Format::Format_RGB32);
    if (newImage.isNull()) {
        QMessageBox::information(this, QGuiApplication::applicationDisplayName(),
                                 tr("Cannot load %1: %2")
                                 .arg(QDir::toNativeSeparators(fileName), reader.errorString()));
        return false;
    }

    setInitialImage(newImage);

    setWindowFilePath(fileName);

    const QString message = tr("Opened \"%1\", %2x%3, Depth: %4")
        .arg(QDir::toNativeSeparators(fileName)).arg(initialImage.width()).arg(initialImage.height()).arg(initialImage.depth());
    statusBar()->showMessage(message);
    return true;
}

void ImageViewer::setInitialImage(const QImage &initialImage)
{
    this->initialImage = initialImage;

    // Invoke a kernel as soon as image is loaded, because first kernel is always slower
    Processor::blurImage(initialImage);

    initialImageLabel->setPixmap(QPixmap::fromImage(initialImage));
    scaleFactor = 1.0;

    initialScrollArea->setVisible(true);
    fitToWindowAct->setEnabled(true);
    updateActions();

    if (!fitToWindowAct->isChecked())
        initialImageLabel->adjustSize();
}

void ImageViewer::setFinalImage(const QImage &finalImage)
{
    this->finalImage = finalImage;
    finalImageLabel->setPixmap(QPixmap::fromImage(finalImage));
    scaleFactor = 1.0;

    finalScrollArea->setVisible(true);
    fitToWindowAct->setEnabled(true);
    updateActions();

    if (!fitToWindowAct->isChecked())
        finalImageLabel->adjustSize();
}


bool ImageViewer::saveFile(const QString &fileName)
{
    QImageWriter writer(fileName);

    if (!writer.write(finalImage)) {
        QMessageBox::information(this, QGuiApplication::applicationDisplayName(),
                                 tr("Cannot write %1: %2")
                                 .arg(QDir::toNativeSeparators(fileName)), writer.errorString());
        return false;
    }
    const QString message = tr("Wrote \"%1\"").arg(QDir::toNativeSeparators(fileName));
    statusBar()->showMessage(message);
    return true;
}


static void initializeImageFileDialog(QFileDialog &dialog, QFileDialog::AcceptMode acceptMode)
{
    static bool firstDialog = true;

    if (firstDialog) {
        firstDialog = false;
        dialog.setDirectory(QDir::currentPath());
    }

    QStringList mimeTypeFilters;
    const QByteArrayList supportedMimeTypes = acceptMode == QFileDialog::AcceptOpen
        ? QImageReader::supportedMimeTypes() : QImageWriter::supportedMimeTypes();
    foreach (const QByteArray &mimeTypeName, supportedMimeTypes)
        mimeTypeFilters.append(mimeTypeName);
    mimeTypeFilters.sort();
    dialog.setMimeTypeFilters(mimeTypeFilters);
    dialog.selectMimeTypeFilter("image/jpeg");
    if (acceptMode == QFileDialog::AcceptSave)
        dialog.setDefaultSuffix("jpg");
}

void ImageViewer::open()
{
    QFileDialog dialog(this, tr("Open File"));
    initializeImageFileDialog(dialog, QFileDialog::AcceptOpen);

    while (dialog.exec() == QDialog::Accepted && !loadFile(dialog.selectedFiles().first())) {}
}

void ImageViewer::saveAs()
{
    QFileDialog dialog(this, tr("Save File As"));
    initializeImageFileDialog(dialog, QFileDialog::AcceptSave);

    while (dialog.exec() == QDialog::Accepted && !saveFile(dialog.selectedFiles().first())) {}
}

void ImageViewer::copy()
{
#ifndef QT_NO_CLIPBOARD
    QGuiApplication::clipboard()->setImage(finalImage);
#endif // !QT_NO_CLIPBOARD
}

#ifndef QT_NO_CLIPBOARD
static QImage clipboardImage()
{
    if (const QMimeData *mimeData = QGuiApplication::clipboard()->mimeData()) {
        if (mimeData->hasImage()) {
            const QImage image = qvariant_cast<QImage>(mimeData->imageData());
            if (!image.isNull())
                return image;
        }
    }
    return QImage();
}
#endif // !QT_NO_CLIPBOARD

void ImageViewer::paste()
{
#ifndef QT_NO_CLIPBOARD
    const QImage newImage = clipboardImage();
    if (newImage.isNull()) {
        statusBar()->showMessage(tr("No image in clipboard"));
    } else {
        setInitialImage(newImage);
        setWindowFilePath(QString());
        const QString message = tr("Obtained image from clipboard, %1x%2, Depth: %3")
            .arg(newImage.width()).arg(newImage.height()).arg(newImage.depth());
        statusBar()->showMessage(message);
    }
#endif // !QT_NO_CLIPBOARD
}

void ImageViewer::zoomIn()
{
    scaleImage(1.25);
}

void ImageViewer::zoomOut()
{
    scaleImage(0.8);
}

void ImageViewer::normalSize()
{
    initialImageLabel->adjustSize();
    scaleFactor = 1.0;
}

void ImageViewer::fitToWindow()
{
    bool fitToWindow = fitToWindowAct->isChecked();
    initialScrollArea->setWidgetResizable(fitToWindow);
    finalScrollArea->setWidgetResizable(fitToWindow);
    if (!fitToWindow)
        normalSize();
    updateActions();
}

void ImageViewer::modifyClicked()
{
    QImage finalImageCPU;
    QImage finalImageCUDA;

    QElapsedTimer timer;
    timer.start();
    switch (modificationComboBox->currentIndex()) {
        case 0:
            finalImageCPU = Processor::blurImageCPU(initialImage);
            break;
        case 1:
            finalImageCPU = Processor::invertImageCPU(initialImage);
            break;
    }
    qint64 nsElapsedCPU = timer.nsecsElapsed();

    timer.start();
    switch (modificationComboBox->currentIndex()) {
        case 0:
            finalImageCUDA = Processor::blurImage(initialImage);
            break;
        case 1:
            finalImageCUDA = Processor::invertImage(initialImage);
            break;
    }
    qint64 nsElapsedCUDA = timer.nsecsElapsed();

    // Ensure the two images are the same
    unsigned long numBytes = finalImageCPU.bytesPerLine()*finalImageCPU.height();
    if (memcmp((void*)finalImageCPU.bits(), (void*)finalImageCUDA.bits(), numBytes) != 0) {
        const QString text = tr("Err: The CPU image does not match the GPU image. Showing the CPU image on the left and the GPU image on the right.");
        timingInfoLabel->setText(text);
        setInitialImage(finalImageCPU);
        setFinalImage(finalImageCUDA);
    }
    else {
        const QString text = tr("CPU took %1 ms\nCUDA took %2 ms").arg(nsElapsedCPU/(float)1e6).arg(nsElapsedCUDA/(float)1e6);
        timingInfoLabel->setText(text);
        setFinalImage(finalImageCUDA);
    }
}

void ImageViewer::about()
{
    QMessageBox::about(this, tr("About Image Viewer"),
            tr("<p>The <b>GPU Image Processing</b> application compares "
               "CPU-based processing techniques to GPU-based techniques.</p> "));
}

void ImageViewer::createActions()
{
    QMenu *fileMenu = menuBar()->addMenu(tr("&File"));

    QAction *openAct = fileMenu->addAction(tr("&Open..."), this, &ImageViewer::open);
    openAct->setShortcut(QKeySequence::Open);

    saveAsAct = fileMenu->addAction(tr("&Save As..."), this, &ImageViewer::saveAs);
    saveAsAct->setEnabled(false);

    fileMenu->addSeparator();

    QAction *exitAct = fileMenu->addAction(tr("E&xit"), this, &QWidget::close);
    exitAct->setShortcut(tr("Ctrl+Q"));

    QMenu *editMenu = menuBar()->addMenu(tr("&Edit"));

    copyAct = editMenu->addAction(tr("&Copy"), this, &ImageViewer::copy);
    copyAct->setShortcut(QKeySequence::Copy);
    copyAct->setEnabled(false);

    QAction *pasteAct = editMenu->addAction(tr("&Paste"), this, &ImageViewer::paste);
    pasteAct->setShortcut(QKeySequence::Paste);

    QMenu *viewMenu = menuBar()->addMenu(tr("&View"));

    zoomInAct = viewMenu->addAction(tr("Zoom &In (25%)"), this, &ImageViewer::zoomIn);
    zoomInAct->setShortcut(QKeySequence::ZoomIn);
    zoomInAct->setEnabled(false);

    zoomOutAct = viewMenu->addAction(tr("Zoom &Out (25%)"), this, &ImageViewer::zoomOut);
    zoomOutAct->setShortcut(QKeySequence::ZoomOut);
    zoomOutAct->setEnabled(false);

    normalSizeAct = viewMenu->addAction(tr("&Normal Size"), this, &ImageViewer::normalSize);
    normalSizeAct->setShortcut(tr("Ctrl+S"));
    normalSizeAct->setEnabled(false);

    viewMenu->addSeparator();

    fitToWindowAct = viewMenu->addAction(tr("&Fit to Window"), this, &ImageViewer::fitToWindow);
    fitToWindowAct->setEnabled(false);
    fitToWindowAct->setCheckable(true);
    fitToWindowAct->setShortcut(tr("Ctrl+F"));

    QMenu *helpMenu = menuBar()->addMenu(tr("&Help"));

    helpMenu->addAction(tr("&About"), this, &ImageViewer::about);
    helpMenu->addAction(tr("About &Qt"), &QApplication::aboutQt);

    connect(modifyButton, SIGNAL(pressed()), this, SLOT(modifyClicked()));
}

void ImageViewer::updateActions()
{
    saveAsAct->setEnabled(!finalImage.isNull());
    copyAct->setEnabled(!finalImage.isNull());
    zoomInAct->setEnabled(!fitToWindowAct->isChecked());
    zoomOutAct->setEnabled(!fitToWindowAct->isChecked());
    normalSizeAct->setEnabled(!fitToWindowAct->isChecked());
}

void ImageViewer::scaleImage(double factor)
{
    scaleFactor *= factor;
    
    if (initialImageLabel->pixmap()) {
        initialImageLabel->resize(scaleFactor * initialImageLabel->pixmap()->size());
        adjustScrollBar(initialScrollArea->horizontalScrollBar(), factor);
        adjustScrollBar(finalScrollArea->verticalScrollBar(), factor);
    }

    if (finalImageLabel->pixmap()) {
        finalImageLabel->resize(scaleFactor * finalImageLabel->pixmap()->size());
        adjustScrollBar(finalScrollArea->horizontalScrollBar(), factor);
        adjustScrollBar(finalScrollArea->verticalScrollBar(), factor);
    }

    zoomInAct->setEnabled(scaleFactor < 3.0);
    zoomOutAct->setEnabled(scaleFactor > 0.333);
}

void ImageViewer::adjustScrollBar(QScrollBar *scrollBar, double factor)
{
    scrollBar->setValue(int(factor * scrollBar->value()
                            + ((factor - 1) * scrollBar->pageStep()/2)));
}