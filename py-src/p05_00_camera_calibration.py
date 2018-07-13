#%% Step 0 - Initialization
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def print_section_header(title):
    """
    Helper function to print section header with given title
    :param title:
    :return:
    """
    print()
    print('#' * 35)
    print('#', title)
    print('#' * 35)

calib_img_dir = '../camera_cal/'
output_dir = '../output_images/calibration/'
results_dir = '../results/'


#%% Step 1 - Analyze calibration image
print_section_header("Calibration Images")
sample_img_file = 'calibration2.jpg'
sample_img_path = calib_img_dir + sample_img_file
sample_img = cv2.imread(sample_img_path)
img_y_size, img_x_size = sample_img.shape[0:2]
print("Sample Image: {}".format(sample_img_path))
print("Image Size: {}x{}".format(img_x_size, img_y_size))
print("Image Min/Max Values: ({}, {})".format(sample_img.min(), sample_img.max()))


#%% Step 2 - Find chessboard corners
print_section_header("Find Chessboard Corners")
nx = 9
ny = 6

# Find object points and image points on each calibration image
objpoints = []
imgpoints = []

# Prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0), ..., (7, 5, 0)
objp = np.zeros((nx * ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

images = sorted(os.listdir(calib_img_dir))
for img_file in images:
    # Read an image file and convert it to grayscale
    img_path = calib_img_dir + img_file
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, imgp = cv2.findChessboardCorners(img_gray, (nx, ny), None)
    if not ret:
        print('Warning: Failed to find chessboard corners for {}'.format(img_file))
    else:
        imgpoints.append(imgp)
        objpoints.append(objp)

nb_calib_img = len(images)
nb_calib_img_valid = len(imgpoints)
print("Calibration Images: {} valid calibration images out of {} ({:.2%})".format(
    nb_calib_img_valid, nb_calib_img, nb_calib_img_valid/nb_calib_img))


#%% Step 3 - Calibrate camera
print_section_header("Calibrate Camera")

# Compute camera matrix and distortion coefficient
retval, camera_mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_x_size, img_y_size), None, None)
print("Re-projection Error: {:.4f}".format(retval))
print("Camera Matrix: {}".format(camera_mtx))
print("Distortion Coefficient: {}".format(dist_coeffs))


#%% Step 4 - Pickle (store) the computed camera matrix and distortion coefficients
print_section_header("Store Calibration Parameters")
import pickle
pickle_file = results_dir + 'camera_cal.p'
camera_cal = {
    'reprojection_error': retval,
    'camera_matrix': camera_mtx,
    'distortion_coefficients': dist_coeffs
}
print("Pickle the camera calibration parameters to {}".format(pickle_file))
with open(pickle_file, 'wb') as outf:
    pickle.dump(camera_cal, outf)


#%% Step 5 - Show undistorted image
print_section_header("Display Undistorted Images")
for img_file in ['calibration1.jpg', 'calibration2.jpg', 'calibration3.jpg']:
    img_name = img_file.split('.')[0]
    img_path = calib_img_dir + img_file
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_undist = cv2.undistort(img, camera_mtx, dist_coeffs, None, camera_mtx)

    # Plot the result
    fig, (sub1, sub2) = plt.subplots(1, 2, figsize=(24, 9))
    fig.tight_layout()
    sub1.imshow(img)
    sub1.set_title("{} (Original)".format(img_file), fontsize=40)
    sub2.imshow(img_undist)
    sub2.set_title("{} (Undistorted)".format(img_file), fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    # Save output files
    outfile = output_dir + 'undistort_' + img_name + '.png'
    print("Store distortion correction images to {}".format(outfile))
    plt.savefig(outfile)


