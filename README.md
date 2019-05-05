# Introduction
This programs implements image processing techniques using OpenMP, CUDA, and OpenCL and then compares their performance.

Specifically, there are two supported operations: image blur and image inversion. The image blur operation is a
simple averaging of all bordering pixels, and the image inversion inverts each individual pixel.
Both of these operations are implemented in OpenMP, CUDA, and OpenCL.


# Screenshots
## Blurred Image
![blurred image](screenshots/blurred-image.png)

## Inverted Image
![inverted image](screenshots/inverted-image.png)

# Build Instructions
To build and run:
- `make`
- `./gpu-image-processing`


# Program Usage
To use the program:
1. `File`->`Open` and select an image. The selected image will be placed on the left.
2. Select the desired image processing operation from the dropdown menu.
3. Click the `Modify` button. This will process the image with each implementation (i.e., OpenMP, CUDA, and OpenCL),
    place the result on the right, and then add text at the bottom of the screen indicating the execution time of each
    implementation.

In addition, the following controls are implemented:
- `Ctrl`+`F` - Fit image to window
- `Ctrl`+`+` - Zoom in (only possible when not fit to window)
- `Ctrl`+`-` - Zoom out (only possible when not fit to window)

For convenience, there are three images included in the `images` directory: `lowres-image.jpg`, `medres-image.jpg`, and `highres-image.jpg`.
The different resolutions are convenient for showing how execution time scales with image resolution. In addition,
images with low resolution are optimal for viewing the effects of the blur operation.

# Acknowledgements:
- The [Image Viewer Example](https://doc.qt.io/qt-5/qtwidgets-widgets-imageviewer-example.html) from Qt was used as the
  basis point for the Qt application. This was extended to support two images, the process effect dropdown, and the modify button;
  in addition, certain functionality was modified to support the new purpose of the application.
- Example code from [this course](http://www.prinmath.com/csci5229/Sp19/description.html) was consulted for the usage of CUDA and OpenCL.
