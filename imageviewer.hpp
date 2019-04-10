#pragma once

#include <QMainWindow>
#include <QGridLayout>
#include <QPushButton>
#include <QComboBox>

class QAction;
class QLabel;
class QMenu;
class QScrollArea;
class QScrollBar;

class ImageViewer : public QMainWindow
{
    Q_OBJECT

public:
    ImageViewer();
    bool loadFile(const QString &);

private slots:
    void open();
    void saveAs();
    void copy();
    void paste();
    void zoomIn();
    void zoomOut();
    void normalSize();
    void fitToWindow();
    void modifyClicked();
    void about();

private:
    void createActions();
    void updateActions();
    bool saveFile(const QString &fileName);
    void setInitialImage(const QImage &initialImage);
    void setFinalImage(const QImage &finalImage);
    void scaleImage(double factor);
    void adjustScrollBar(QScrollBar *scrollBar, double factor);

    QWidget *centralWidget;
    QGridLayout *gridLayout;

    QComboBox *modificationComboBox;
    QPushButton *modifyButton;
    QLabel *timingInfoLabel;

    double scaleFactor;

    QImage initialImage;
    QLabel *initialImageLabel;
    QScrollArea *initialScrollArea;

    QImage finalImage;
    QLabel *finalImageLabel;
    QScrollArea *finalScrollArea;

    QAction *saveAsAct;
    QAction *copyAct;
    QAction *zoomInAct;
    QAction *zoomOutAct;
    QAction *normalSizeAct;
    QAction *fitToWindowAct;
};