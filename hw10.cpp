//
//  Hw10:  OpenCL
//

#include <QApplication>
#include "imageviewer.hpp"

//
//  Main function
//
int main(int argc, char *argv[])
{
   //  Create the application
   QApplication app(argc,argv);
   //  Create and show view widget
   ImageViewer view;
   view.show();
   //  Main loop for application
   return app.exec();
}
