# 3D Reconstruction

Implantation of Stereo Camera Calibration and 3D reconstruction using Epipolar Geometry.

## Implementation
This project is an implementation from scratch. There is also an implementation using OpenCV's function for comparison.

You can find the theory relevant to this project in the following link:
- [Elements of Geometry for Computer Vision and Computer Graphics](https://cw.fel.cvut.cz/wiki/_media/courses/gvg/pajdla-gvg-lecture-2021.pdf)


## Data
The images used in this project were taken using another project of mine : [LensDistortionOpenGL](https://github.com/Vlhermitte/LensDistortionOpenGL)

It is assumed that the images are already undistorted and that we know the FOV of the cameras which allows us to calculate the intrinsic matrices K.

---
## Requirements
- OpenCV
- Numpy
- Matplotlib
- Scipy