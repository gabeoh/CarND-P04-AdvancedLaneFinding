# Advanced Lane Finding

## Objective

The goal of this project is to identify lane boundaries from a front-facing
camera on a vehicle dashboard.  The following steps are required in order to
successfully meet the project goal.

1. Correct distorted raw camera images using computed camera calibration matrix
   and distortion coefficients
1. Create a binary image highlighting lane lines by using color transform and
   gradients
1. Transform front-facing camera images into top-down (bird's eye) view images
   by applying perspective transform
1. Detect lane pixels and find fitting polynomials for lane lines
1. Overlay detected lane boundaries to the original image applying inverse
   perspective transform
1. Output visual display of lane boundaries and numerical estimation of lane
   curvature and vehicle position


[//]: # (Image References)

[img_calib_01]: ./output_images/calibration/undistort_calibration1.png
[img_calib_02]: ./output_images/calibration/undistort_calibration2.png
[img_calib_03]: ./output_images/calibration/undistort_calibration3.png
[img_test_03]: ./test_images/test3.jpg
[img_undist_test_03]: ./output_images/undistorted/test3.png
[img_bin_lane_03]: ./output_images/binary_lanes/test3.png
[img_pt_src_sl_01]: ./output_images/perspective/marked/straight_lines1_src.png
[img_pt_dst_sl_01]: ./output_images/perspective/marked/straight_lines1.png
[img_pt_src_test_05]: ./output_images/perspective/marked/test5_src.png
[img_pt_dst_test_05]: ./output_images/perspective/marked/test5.png
[img_pt_src_bin_test_03]: ./output_images/perspective/marked_bin/test3_src.png
[img_pt_dst_bin_test_03]: ./output_images/perspective/marked_bin/test3.png
[img_slide_win_test_03]: ./output_images/slide_win/test3.png
[img_overlay_test_03]: ./output_images/overlay/test3.png


---
## Camera Calibration

### 1. Camera Matrix and Distortion Coefficients
The source code for the camera calibration is located in
[py-src/p05_00_camera_calibration.py](py-src/p05_00_camera_calibration.py).

#### Find chessboard corners _`(Line: 35-66)`_
First, _object points_ are prepared for the chessboard with 9x6 inner corners.
Then, _image points_ are gathered from 17 of 20 provided camera calibration
images using `cv2.findChessboardCorners()` function.  It failed to find
chessboard corners for three images (`calibration[1,4,5].jpg`) as the images
do not capture all 54 required inner corners.

#### Compute camera matrix and distortion coefficient _`(Line: 69-76)`_
The `camera_mtx` and `dist_coeffs` are computed using
`cv2.calibrateCamera()` function with `objpoints` and `imgpoints` collected
in the above step.

#### Undistort Images _`(Line: 93-114)`_
The distortion correction is carried out by applying `cv2.undistort()`
function using camera matrix and distortion coefficient computed above.



### 2. Distortion Correction on Calibration Images
Performed the distortion correction on three of the provided calibration
images (calibration1.jpg, calibration2.jpg, and calibration3.jpg).  The result
of the distortion correction is demonstrated below
([output_images/calibration](output_images/calibration)).

![undistort_calibration1.png][img_calib_01]
![undistort_calibration2.png][img_calib_02]
![undistort_calibration3.png][img_calib_03]


---
## Lane Detection on Test Images

### 1. Distortion Correction on Test Images
Image distortion correction is performed on provided test images using
camera matrix and distortion coefficients computed in the camera calibration
step.  The corrected images are stored in
[output_images/undistorted](output_images/undistorted)
directory.

As shown in the example below, the distortion correction is prominent for the
horizontal line near the bottom of the image.

#### Original Test Image (test3.jpg)
![test3.jpg][img_test_03]

#### Undistorted Test Image (test3.png)
![test3.png][img_undist_test_03]


### 2. Create Lane Pixel Images
The source code for creating lane pixel images is in
[py-src/p05_02_create_binary_lane.py](py-src/p05_02_create_binary_lane.py).

A combination of gradient and color thresholds is used to create binary
image files that contain lane pixels.
Three threshold methods are used to identify lane pixel candidates.  The end
result is the union of pixels from each of the three methods.

- Gradient in X-Direction
  - `get_binary_lane_pixels_x_grad()`
- Gradient Direction and Magnitude
  - `get_binary_lane_pixels_grad_dir_mag()`
- Color Threshold on Saturation Channel
  - `get_binary_lane_pixels_color_saturation()`

The table below shows threshold values used for each threshold method.
There is no definite rules in determining these threshold values.
The values are determined by iterative experiments and visual verifications.

| Type                 | Threshold Min/Max  | 
|:--------------------:|:------------------:| 
| X-Gradient           | 40, 150            | 
| Gradient Direction   | 0.3, 1.25          |
| Gradient Magnitude   | 45, 150            |
| S-Channel Color      | 170, 255           |


The resulting binary lane pixel images are stored in
[output_images/binary_lanes](output_images/binary_lanes) directory.
An example image is shown below.

#### Binary Lane Image (test3.png)
![test3.png][img_bin_lane_03]


### 3. Perspective Transform
The source code for the perspective transform is located in
[py-src/p05_03_perspective_transform.py](py-src/p05_03_perspective_transform.py)

First, the source corners for the perspective transform are found using
undistorted version of the two straight lane line test images
(`straight_lines1.png` and `straight_lines2.png`).  The coordinates of
the corresponding destination corners are determined to place the area of
interest in the middle of the transformed image.

The table below shows the coordinates of source and destination corners:

| Corner        | Source        | Destination   |
|:-------------:|:-------------:|:-------------:|
| Top Left      | 593, 450      | 320, 0        |
| Top Right     | 687, 450      | 960, 0        |
| Bottom Right  | 1100, 720     | 960, 720      |
| Bottom Left   | 180, 720      | 320, 720      |

With these source and destination corner coordinates, the perspective
transform matrix (and it inverse transform) is computed using
`cv2.getPerspectiveTransform()` function.

The computed transform matrix is verified on test images.
Transformed images with source and destination corner markings are located
under 
[output_images/perspective/](output_images/perspective/) directory.
Examples of the transformation are shown below.

#### Perspective Transform of Straight Lane Image
**Original Image (undistorted)** - _straight_lines1_src.png_
![straight_lines1_src.png][img_pt_src_sl_01]

**Transformed Image** - _straight_lines1.png_
![straight_lines1.png][img_pt_dst_sl_01]

#### Perspective Transform of Curved Lane Image
**Original Image (undistorted)** - _test5_src.png_
![test5_src.png][img_pt_src_test_05]

**Transformed Image** - _test5.png_
![test5.png][img_pt_dst_test_05]

#### Perspective Transform of Binaray Lane Image
**Original Image (lane pixels)** - _test3_src.png_
![test3_src.png][img_pt_src_bin_test_03]

**Transformed Image** - _test3.png_
![test3.png][img_pt_dst_bin_test_03]


### 4. Identify Lane-Line Pixels
The source code for the lane-line pixel identification is located in
[py-src/p05_04_identify_lane_line.py](py-src/p05_04_identify_lane_line.py).

Histogram and sliding window methods are used to identify lane-line pixels
in the test images.

First, the base x coordinates for left and right lane
lines are determined using the histogram on the bottom half of the image
(x coordinates with maximum non-zero pixel occurrence).

Then, the image is divided into 9 vertical sections for the sliding window
method.  The x coordinates from histogram are used as initial window center
for the bottom window.  Subsequent window centers are set to the mean of
valid pixels' x-coordinates in the previous window by sliding windows from
the bottom to the top of the test images.

Once all lane line pixels are identified, the 2nd order polynomial is used
to characterize the lane lines. 

The lane line identified images are located
under 
[output_images/slide_win/](output_images/slide_win/) directory.
An example image is shown below.

**Lane Detected Image** - _test3.png_
- f(y)<sub>left</sub> = 0.000296y<sup>2</sup> - 0.5442y + 597.9
- f(y)<sub>right</sub> = 0.000295y<sup>2</sup> - 0.4982y + 1193.1
![test3.png][img_slide_win_test_03]


### 5. Calculate Lane Curvature and Vehicle Position
The implementation of calculating lane curvature and vehicle position
is located in
[py-src/p05_05_curvature_offset.py](py-src/p05_05_curvature_offset.py).

The radius of lane curvature is calculated in meters at the bottom of
the image.  In assumption of the lane width of 3.7m (over 640 pixels) and
the length of 30m (over 720 pixels), 3.7/640 and 30/720 meters
per pixel converting factors are used for x and y dimensions respectively. 

With pixel-to-meter converting factors, the polynomial coefficients and
the evaluating y-coordinate in pixels are converted to in meters.
Then, the lane curvature is computed using _radius of curvature_ formula.

The vehicle position (ie. horizontal offset from lane center in meters)
is determined by computing the distance between image center
(vehicle position) and the lane mid point.
Negative offsets indicate that the vehicle is positioned on the left of
the lane center, and positive ones indicate the vehicle in on the right.

The table below shows computed lane curvature and vehicle offset of
each test image.

| Image File        | Curvature Left  | Curvature Right | Position Offset |
|:-----------------:|:---------------:|:---------------:|:---------------:|
| straight_lines1   | 5867.5 m        | 1422.9 m        | -0.035 m        |
| straight_lines2   | 1511.9 m        | 3567.5 m        | -0.063 m        |
| test1             | 371.0 m         | 341.0 m         | -0.279 m        |
| test2             | 379.0 m         | 310.0 m         | -0.377 m        |
| test3             | 507.4 m         | 509.3 m         | -0.193 m        |
| test4             | 776.1 m         | 142.2 m         | -0.497 m        |
| test5             | 398.4 m         | 242.2 m         | -0.119 m        |
| test6             | 1185.5 m        | 1292.0 m        | -0.262 m        |


### 6. Overlay Detected Lane Lines on Original Image
The source code for the lane area overlaying is located in
[py-src/p05_06_overlay_annotate.py](py-src/p05_06_overlay_annotate.py).

The lane area and lane line pixels identified from previous steps are filled
with distinct colors (green for lane area, red and blue for left and right 
lane lines respectively).  Then, the highlight image is warped back to the
original image perspective and overlayed onto the undistorted version of the
original image.

In addition, the radius of the lane curvature and vehicle offset are included
in the resulting images.

The original images (undistorted) with lane area highlighted are located under
[output_images/overlay/](output_images/overlay/) directory.
An example image is shown below.

**Lane Overlay Image** - _test3.png_
![test3.png][img_overlay_test_03]


---
## Lane Detection on Video

### 1. Apply Lane Detection on Video
This a link to the final video output:
- [project_video.mp4](./output_images/video/project_video.mp4)


### 2. Challenging Videos
I tried running the pipeline on two provided challenging videos.
Unfortunately, the pipeline did not perform well on these videos.
The implementation limitation and potential improvements are discussed in the
following section.

These are links to the results from challenging videos:
- [challenge_video.mp4](./output_images/video/challenge_video.mp4)
- [harder_challenge_video.mp4](./output_images/video/harder_challenge_video.mp4)


---
## Discussion

### 1. Limitation and Future Works

#### Lane Line Pixel Detection Error
One of the most prominent problems I noticed in the challenging videos is
that the incorrect recognition of lane line pixels.  For example, in
`challenge_video.mp4`, the road divider exists on the far left of the
left lane lines.  However, the pipeline often confused the shadow boundaries
of the divider as lane lines.

Similarly, certain road pavement irregularity also contributed the the
detection error.  Some parts of `challenge_video.mp4` has lane pavement
shade differences.  Especially when this boundary runs in parallel direction
with lane lines in their proximity, the pipeline occasionally confuses the
boundaries as lane lines.

This is a limitation of the gradient based lane line pixel identification
in step 2.  More studies on the characteristics of lane lines and other common
distractors are required to improve this pipeline step.

#### Road Slopes
Another challenge is with road slopes.  The perspective transform algorithm
used in the pipeline assumes that the road surface is relatively flat.
Therefore, for the road with higher slope angles such as in
`harder_challenge_video.mp4`, the pipeline did not perform ideally. 

#### Steep Curves
The `harder_challenge_video.mp4` also contains steep curves where one of the
lane lines completely disappears from the view.  The implemented lane line
recognition algorithm assumes two lane lines on the left and the right
from the center.  Such steep curves break this assumption and causes the
pipeline to behave inaccurately.

A similar problem can occur when the vehicle goes too far from the center
of the lane although the test video did not exhibit such behavior.
Even if the lane lines do not completely disappear from the view, it is
possible that they are not fully represented in the perspective transformed
images.

#### Other Potential Improvements
Further improvement can be made to the pipeline by considering cross-frame
relations.  By utilizing results (such as polynomial fit) from previous frames, 
the pipeline can improve its performance (speed).  This information can also
be used to reject outliers, which would reduce jitters and smoothen lane
detections.


---
## Appendix
### 1. Source and Outputs
#### Source Code
- Camera Calibration 
  - [p05_01_camera_calibration.py](py-src/p05_00_camera_calibration.py)
- Lane Detection 
  - [p05_lane_detection_main.py](py-src/p05_lane_detection_main.py)
  - [p05_01_correct_distortion.py](py-src/p05_01_correct_distortion.py)
  - [p05_02_create_binary_lane.py](py-src/p05_02_create_binary_lane.py)
  - [p05_03_perspective_transform.py](py-src/p05_03_perspective_transform.py)
  - [p05_04_identify_lane_line.py](py-src/p05_04_identify_lane_line.py)
  - [p05_05_curvature_offset.py](py-src/p05_05_curvature_offset.py)
  - [p05_06_overlay_annotate.py](py-src/p05_06_overlay_annotate.py)

#### Execution Log
- Camera Calibration 
  - [camera_calibration.log](results/camera_calibration.log)
- Lane Detection (test images) 
  - [lane_detection_test_images.log](results/lane_detection_test_images.log)

#### Output Images
- Camera Calibration
  - [output_images/calibration/](output_images/calibration/)
- Distortion Correction
  - [output_images/undistorted/](output_images/undistorted/)
- Binary Lane Pixels
  - [output_images/binary_lanes/](output_images/binary_lanes/)
- Perspective Transform
  - [output_images/perspective/](output_images/perspective/)
- Lane Identified Images (sliding windows)
  - [output_images/slide_win/](output_images/slide_win/)
- Lane Overlay
  - [output_images/overlay/](output_images/overlay/)
  
#### Output Videos
- Videos
  - [output_images/video/](output_images/video/)