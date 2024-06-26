# 3D Reconstruction

Implantation of Stereo Camera Calibration and 3D reconstruction using Epipolar Geometry.

## Implementation
This project is an implementation from scratch. There is also an implementation using OpenCV's function for comparison.

The project is divided into 4 parts:
1. **Calibration**: 
    - Calculate the intrinsic matrices K of the cameras using the FOV.
2. **Local features matching**:
    - Use SIFT to find the keypoints and descriptors of the images.
    - Use the descriptors to find the matches between the images.
3. **Epipolar Geometry**:
    - Calculate the fundamental matrix F using the matches and LO-RANSAC.
    - Calculate the essential matrix E using the fundamental matrix and the intrinsic matrices.
    - Calculate the rotation and translation matrices using the essential matrix.
    - Triangulate the points using the rotation and translation matrices.
4. **3D Reconstruction**:
   - Use the triangulated points to reconstruct the 3D scene.
   - Perform 3D mesh reconstruction using convex hull. (Not implemented yet)

## Data
The images used in this project were taken using another project of mine : [LensDistortionOpenGL](https://github.com/Vlhermitte/LensDistortionOpenGL)

It is assumed that we already know the FOV of the cameras which allows us to calculate the intrinsic matrices K.

## Results
The results are shown in the following images:

OpenCV's implementation vs Custom implementation:

<img src="Results/Epipolar%20lines%20(OpenCV).png" width="384">   <img src="Results/Epipolar%20lines%20(Custom).png" width="384">

## Possible improvements
- Use a better feature matching algorithm. (Maybe a CNN)
- Implement the 3D mesh reconstruction.

## References
You can find the theory relevant to this project in the following link:
- [Elements of Geometry for Computer Vision and Computer Graphics](https://cw.fel.cvut.cz/wiki/_media/courses/gvg/pajdla-gvg-lecture-2021.pdf)
- [LO-RANSAC](http://cmp.felk.cvut.cz/software/LO-RANSAC/Lebeda-2012-Fixing_LORANSAC-BMVC.pdf)

---
## Dependencies
- OpenCV
- Numpy
- Matplotlib
- Scipy