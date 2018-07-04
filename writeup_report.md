# Advanced Lane Finding

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car.

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


---
## Camera Calibration

### 1. Camera Matrix and Distortion Coefficients
The source code for the camera calibration is located in
[py-src/p05_01_camera_calibration.py](py-src/p05_01_camera_calibration.py).

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

#### Undistort Images _`(Line: 79-98)`_
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
The source code for creating lane pixel images is located at
`Line 66-141` of 
[py-src/p05_02_lane_detection.py](py-src/p05_02_lane_detection.py).

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
The source code for the perspective transform is located at
`Line 144-231` of 
[py-src/p05_02_lane_detection.py](py-src/p05_02_lane_detection.py).

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
The source code for the lane-line pixel identification is located at
`Line 234-362` of 
[py-src/p05_02_lane_detection.py](py-src/p05_02_lane_detection.py).

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
under `output_images/slide_win/` directory.
An example image is shown below.

**Lane Detected Image** - _test3.png_
- f(y)<sub>left</sub> = 0.000296y<sup>2</sup> - 0.5442y + 597.9
- f(y)<sub>right</sub> = 0.000295y<sup>2</sup> - 0.4982y + 1193.1
![test3.png][img_slide_win_test_03]


### 5. Calculate Lane Curvature and Vehicle Position
**Describe how (and identify where in your code) you calculated the radius 
of curvature of the lane and the position of the vehicle with respect to 
center.**

_Here the idea is to take the measurements of where the lane lines are and 
estimate how much the road is curving and where the vehicle is located with 
respect to the center of the lane. The radius of curvature may be given in 
meters assuming the curve of the road follows a circle. For the position of 
the vehicle, you may assume the camera is mounted at the center of the car 
and the deviation of the midpoint of the lane from the center of the image 
is the offset you're looking for. As with the polynomial fitting, convert 
from pixels to meters._



### 6. Overlay Detected Lane Lines on Original Image
**Provide an example image of your result plotted back down onto the road 
such that the lane area is identified clearly.**

_The fit from the rectified image has been warped back onto the original 
image and plotted to identify the lane boundaries. This should demonstrate 
that the lane boundaries were correctly identified. An example image with 
lanes, curvature, and position from center should be included in the writeup 
(or saved to a folder) and submitted with the project._

I implemented this step in lines # through # in my code in 
`yet_another_file.py` in the function `map_lane()`.  Here is an example 
of my result on a test image:

![alt text][image6]


---
## Lane Detection on Video

### 1. Apply Lane Detection on Video
**Provide a link to your final video output. Your pipeline should perform 
reasonably well on the entire project video (wobbly lines are ok but no 
catastrophic failures that would cause the car to drive off the road!)**

_The image processing pipeline that was established to find the lane lines 
in images successfully processes the video. The output here should be a new 
video where the lanes are identified in every frame, and outputs are 
generated regarding the radius of curvature of the lane and vehicle position 
within the lane. The pipeline should correctly map out curved lines and not 
fail when shadows or pavement color changes are present. The output video 
should be linked to in the writeup and/or saved and submitted with the project._

Here's a [link to my video result](./project_video.mp4)


### 2. Utilize Previous Detection _(Standout)_
_For a standout submission, you should follow the suggestion in the lesson to 
not just search blindly for the lane lines in each frame of video, but rather, 
once you have a high-confidence detection, use that to inform the search for 
the position of the lines in subsequent frames of video. For example, if a 
polynomial fit was found to be robust in the previous frame, then rather than 
search the entire next frame for the lines, just a window around the previous 
detection could be searched. This will improve speed and provide a more robust 
method for rejecting outliers._

### 3. Outlier Rejection _(Standout)_
_For an additional improvement you should implement outlier rejection and use 
a low-pass filter to smooth the lane detection over frames, meaning add each 
new detection to a weighted mean of the position of the lines to avoid jitter._

### 4. Challenging Video _(Standout)_
_If you really want to go above and beyond, implement these methods on the 
challenge videos as well, or on your own videos you've recorded yourself._


---
## Discussion

### 1. Limitation and Future Works
**Briefly discuss any problems / issues you faced in your implementation of 
this project. Where will your pipeline likely fail? What could you do to make 
it more robust?**

_Discussion includes some consideration of problems/issues faced, what 
could be improved about their algorithm/pipeline, and what hypothetical 
cases would cause their pipeline to fail._

Here I'll talk about the approach I took, what techniques I used, what 
worked and why, where the pipeline might fail and how I might improve it if 
I were going to pursue this project further.  


---
## Appendix
### 1. Source and Outputs
#### Source Code
- Camera Calibration 
  - [p05_01_camera_calibration.py](py-src/p05_01_camera_calibration.py)
- Lane Detection 
  - [p05_02_lane_detection.py](py-src/p05_02_lane_detection.py)

#### Execution Log
- Camera Calibration 
  - [camera_calibration.log](results/camera_calibration.log)
- Lane Detection (test images) 
  - [lane_detection_test_images.log](results/lane_detection_test_images.log)

#### Output Images
- Camera Calibration
  - [output_images/calibration](output_images/calibration)
- Distortion Correction
  - [output_images/undistorted](output_images/undistorted)
- Binary Lane Pixels
  - [output_images/binary_lanes](output_images/binary_lanes)
- Perspective Transform
  - [output_images/perspective](output_images/perspective)
