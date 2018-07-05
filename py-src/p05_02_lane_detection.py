#%% Initialization
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle
import argparse

output_dir = '../output_images/'
results_dir = '../results/'
test_img_dir = '../test_images/'
undistorted_img_dir = output_dir + 'undistorted/'
binary_lane_dir = output_dir + 'binary_lanes/'
perspective_trans_dir = output_dir + 'perspective/'
slide_win_dir = output_dir + 'slide_win/'

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


#%% Step 0 - Analyze test image
def analyze_test_image(img_path):
    print_section_header("Test Images")
    img = cv2.imread(img_path)
    img_y_size, img_x_size = img.shape[0:2]
    print("Sample Image: {}".format(img_path))
    print("Image Size: {}x{}".format(img_x_size, img_y_size))
    print("Image Min/Max Values: ({}, {})".format(img.min(), img.max()))


#%% Step 1 - Correct image distortion
def correct_image_distortion(pickle_file):
    print_section_header("Correct Image Distortions")

    # Load camera calibration parameters
    with open(pickle_file, 'rb') as inf:
        camera_cal = pickle.load(inf)
    camera_mtx = camera_cal['camera_matrix']
    dist_coeffs = camera_cal['distortion_coefficients']

    images = sorted(os.listdir(test_img_dir))
    images = [f for f in images if not f.startswith('.')]
    for img_file in images:
        # Read an image file and correct distortions
        img_name = img_file.split('.')[0]
        img_path = test_img_dir + img_file
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_undist = cv2.undistort(img, camera_mtx, dist_coeffs, None, camera_mtx)

        # Save output files
        outfile = undistorted_img_dir + img_name + '.png'
        print("Store the undistorted image to {}".format(outfile))
        cv2.imwrite(outfile, cv2.cvtColor(img_undist, cv2.COLOR_RGB2BGR))


#%% Step 2 - Create binary lane pixel images using color and gradient thresholds
def get_binary_lane_pixels_x_grad(img, thresh_sobel_x=(40, 150), sobel_kernel=15):
    # 1) Convert the image to greyscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    # 3) Create a binary masks for sobel_x
    sobel_x_abs = np.abs(sobel_x)
    sobel_x_scaled = np.array((255.0 / sobel_x_abs.max()) * sobel_x_abs, np.uint8)
    bin_x_grad = np.zeros_like(sobel_x_scaled)
    bin_x_grad[(sobel_x_scaled > thresh_sobel_x[0]) & (sobel_x_scaled <= thresh_sobel_x[1])] = 1
    return bin_x_grad

def get_binary_lane_pixels_grad_dir_mag(img, thresh_grad_dir=(0.3, 1.25), thresh_sobel_mag=(45, 150),
                                        sobel_kernel=15):
    # 90 deg = pi/2 = 1.5708
    # 60 deg = pi/3 = 1.0472
    # 45 deg = pi/4 = 0.7854
    # 30 deg = pi/6 = 0.5236

    # 1) Convert the image to greyscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y and get their absolute values
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_x_abs = np.abs(sobel_x)
    sobel_y_abs = np.abs(sobel_y)

    # 3) Create a binary masks for gradient direction and magnitude
    grad_dirs = np.arctan2(sobel_y_abs, sobel_x_abs)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag_scaled = np.array((255.0 / sobel_mag.max()) * sobel_mag, np.uint8)
    bin_grad_dir_mag = np.zeros_like(grad_dirs, np.uint8)
    bin_grad_dir_mag[
        (grad_dirs > thresh_grad_dir[0]) & (grad_dirs <= thresh_grad_dir[1]) &
        (sobel_mag_scaled > thresh_sobel_mag[0]) & (sobel_mag_scaled <= thresh_sobel_mag[1])] = 1
    return bin_grad_dir_mag

def get_binary_lane_pixels_color_saturation(img, thresh_channel_s=(170, 255)):

    # 1) Convert to HLS color space and separate out S channel
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_s = img_hls[:, :, 2]

    # 2) Create a binary masks for S-channel
    bin_saturation = np.zeros_like(img_s)
    bin_saturation[(img_s > thresh_channel_s[0]) & (img_s <= thresh_channel_s[1])] = 1
    return bin_saturation

# Create a binary image with identified lane pixels using color and gradient thresholds
def create_lane_pixel_images():
    print_section_header("Color Transform and Gradients")

    images = sorted(os.listdir(undistorted_img_dir))
    images = [f for f in images if not f.startswith('.')]
    for img_file in images:
        img_name = img_file.split('.')[0]
        img_path = undistorted_img_dir + img_file
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bin_x_grad = get_binary_lane_pixels_x_grad(img)
        bin_grad_dir_mag = get_binary_lane_pixels_grad_dir_mag(img)
        bin_saturation = get_binary_lane_pixels_color_saturation(img)

        bin_combined = np.zeros_like(bin_x_grad)
        bin_combined[(bin_x_grad == 1) | (bin_grad_dir_mag == 1) | (bin_saturation == 1)] = 1
        img_bin_lane = bin_combined * 255

        # Save output files
        outfile = binary_lane_dir + img_name + '.png'
        print("Store the binary lane image to {}".format(outfile))
        cv2.imwrite(outfile, img_bin_lane)


#%% Step 3 - Transform image perspective
# Determine perspective transform source and destination corners
def determine_perspective_transform_corners():
    # Determine source and destination corners from straight line images
    img_height = 720
    src_top, src_bot = 450, img_height
    src_top_l, src_top_r = 593, 687
    src_bot_l, src_bot_r = 180, 1100
    src_corners = np.array([
        [src_top_l, src_top],
        [src_top_r, src_top],
        [src_bot_r, src_bot],
        [src_bot_l, src_bot],
    ], np.int32)

    dst_top, dst_bot = 0, img_height
    dst_l, dst_r = 320, 960
    dst_corners = np.array([
        [dst_l, dst_top],
        [dst_r, dst_top],
        [dst_r, dst_bot],
        [dst_l, dst_bot],
    ], np.int32)

    return src_corners, dst_corners

# Compute the transform matrix
def compute_perspective_transform_matrix():
    # Compute transform matrices using cv2.getPerspectiveTransform()
    src_corners, dst_corners = determine_perspective_transform_corners()
    src_corners_fl = src_corners.astype(np.float32)
    dst_corners_fl = dst_corners.astype(np.float32)
    mtx_trans = cv2.getPerspectiveTransform(src_corners_fl, dst_corners_fl)
    mtx_trans_inv = cv2.getPerspectiveTransform(dst_corners_fl, src_corners_fl)
    return mtx_trans, mtx_trans_inv

# Demonstrate image perspective transform
def demonstrate_perspective_transform(img_dir, img_file, mtx_trans, output_dir=None, mark_guideline=False):
    img_path = img_dir + img_file
    img_name = img_file.split('.')[0]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform image perspective
    img_height, img_width = img.shape[0:2]
    img_trans = cv2.warpPerspective(img, mtx_trans, (img_width, img_height), flags=cv2.INTER_LINEAR)

    src_corners, dst_corners = None, None
    if mark_guideline:
        src_corners, dst_corners = determine_perspective_transform_corners()

    # Mark the transform guide-line if dst_corners is given
    outfile = output_dir + img_name + '.png'
    if (dst_corners is not None):
        dst_pts = dst_corners.reshape((-1, 1, 2))
        img_trans = np.copy(img_trans)
        cv2.polylines(img_trans, [dst_pts], True, (255, 0, 0), 3)

    print("Store the perspective transform result image to {}".format(outfile))
    cv2.imwrite(outfile, cv2.cvtColor(img_trans, cv2.COLOR_RGB2BGR))

    # Generate transform guide-line marked image for the original image if src_corners is given
    if (src_corners is not None):
        src_pts = src_corners.reshape((-1, 1, 2))
        img_marked = np.copy(img)
        cv2.polylines(img_marked, [src_pts], True, (255, 0, 0), 3)

        outfile = output_dir + img_name + '_src.png'
        print("Store the perspective transform source image to {}".format(outfile))
        cv2.imwrite(outfile, cv2.cvtColor(img_marked, cv2.COLOR_RGB2BGR))

def perform_perspective_transforms():
    print_section_header("Transform Image Perspective")

    # Get matrices for perspective transform and its reverse operation
    mtx_trans, mtx_trans_inv = compute_perspective_transform_matrix()

    # Demonstrate perspective transform
    img_dirs = [undistorted_img_dir, binary_lane_dir, binary_lane_dir]
    marks = [True, True, False]
    output_dirs = [perspective_trans_dir + 'marked/', perspective_trans_dir + 'marked_bin/', perspective_trans_dir]
    for img_dir, mark_guideline, output_dir in zip(img_dirs, marks, output_dirs):
        images = sorted(os.listdir(img_dir))
        images = [f for f in images if not f.startswith('.')]
        for img_file in images:
            demonstrate_perspective_transform(img_dir, img_file, mtx_trans, output_dir, mark_guideline)

    return mtx_trans, mtx_trans_inv


#%% Step 4 - Identify lane line pixels
# Find lane line pixel coordinate indices through sliding windows
def find_lane_pixel_coordinate_indices(img_height, pix_coord_x, pix_coord_y, x_base_l, x_base_r):
    # Number of sliding windows
    n_windows = 9
    # Sliding window width: +/- wind_margin
    win_margin = 100
    # Minimum number of pixels to re-center sliding window
    th_recenter_min_pix = 50

    inds_lane_pix_l, inds_lane_pix_r = [], []
    slide_wins = []
    win_height = img_height // n_windows
    x_curr_l, x_curr_r = x_base_l, x_base_r
    for i_win in range(n_windows):
        # Identify window boundaries
        win_y_low = img_height - (i_win + 1) * win_height
        win_y_high = img_height - i_win * win_height
        win_x_low_l, win_x_high_l = x_curr_l - win_margin, x_curr_l + win_margin
        win_x_low_r, win_x_high_r = x_curr_r - win_margin, x_curr_r + win_margin
        win_l = ((win_x_low_l, win_y_low), (win_x_high_l, win_y_high))
        win_r = ((win_x_low_r, win_y_low), (win_x_high_r, win_y_high))
        slide_wins.append(win_l)
        slide_wins.append(win_r)

        # Find indices of the lane pixels in the window
        inds_pix_in_win_l = ((pix_coord_x >= win_x_low_l) & (pix_coord_x < win_x_high_l)
                             & (pix_coord_y >= win_y_low) & (pix_coord_y < win_y_high)).nonzero()[0]
        inds_pix_in_win_r = ((pix_coord_x >= win_x_low_r) & (pix_coord_x < win_x_high_r)
                             & (pix_coord_y >= win_y_low) & (pix_coord_y < win_y_high)).nonzero()[0]

        # Collect indices of lane line pixels
        inds_lane_pix_l.extend(inds_pix_in_win_l)
        inds_lane_pix_r.extend(inds_pix_in_win_r)

        # Re-center sliding windows if more pixels than minimum pixel threshold are found
        if len(inds_pix_in_win_l) > th_recenter_min_pix:
            pix_coord_x_in_win_l = pix_coord_x[inds_pix_in_win_l]
            x_curr_l = np.int(np.mean(pix_coord_x_in_win_l))
        if len(inds_pix_in_win_r) > th_recenter_min_pix:
            pix_coord_x_in_win_r = pix_coord_x[inds_pix_in_win_r]
            x_curr_r = np.int(np.mean(pix_coord_x_in_win_r))
    return inds_lane_pix_l, inds_lane_pix_r, slide_wins

def find_lane_pixel_coordinates(img):
    # Convert img to binary
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = np.zeros_like(img_gray)
    img_bin[img_gray > img_gray.max() // 2] = 1
    img_height, img_width = img_bin.shape

    # Find x and y coordinates of nonzero pixels
    pix_coord_y, pix_coord_x = img_bin.nonzero()

    # Find x coordinate bases for lines using histogram of the bottom half of the image
    mid_x, mid_y = img_width // 2, img_height // 2
    hist_x = np.sum(img_bin[mid_y:, :], axis=0)
    x_base_l = np.argmax(hist_x[:mid_x])
    x_base_r = np.argmax(hist_x[mid_x:]) + mid_x

    # Find lane pixel coordinate indices
    inds_lane_pix_l, inds_lane_pix_r, slide_wins = find_lane_pixel_coordinate_indices(
        img_height, pix_coord_x, pix_coord_y, x_base_l, x_base_r)

    # Collect all x and y coordinates of both left and right lane lines
    coord_x_lane_l = pix_coord_x[inds_lane_pix_l]
    coord_y_lane_l = pix_coord_y[inds_lane_pix_l]
    coord_x_lane_r = pix_coord_x[inds_lane_pix_r]
    coord_y_lane_r = pix_coord_y[inds_lane_pix_r]
    return (coord_x_lane_l, coord_y_lane_l), (coord_x_lane_r, coord_y_lane_r), slide_wins

def find_fitting_polynomials(img):
    # Find lane pixel coordinates
    coords_lane_l, coords_lane_r, slide_wins = find_lane_pixel_coordinates(img)
    coord_x_lane_l, coord_y_lane_l = coords_lane_l[0], coords_lane_l[1]
    coord_x_lane_r, coord_y_lane_r = coords_lane_r[0], coords_lane_r[1]

    # Fit a second order polynomial to each lane line
    # x = f(y) to avoid fitting error for vertical line
    poly_fit_l = np.polyfit(coord_y_lane_l, coord_x_lane_l, 2)
    poly_fit_r = np.polyfit(coord_y_lane_r, coord_x_lane_r, 2)
    return poly_fit_l, poly_fit_r, coords_lane_l, coords_lane_r, slide_wins

def perform_lane_line_detection_images(img_dir, out_dir=None):
    print_section_header("Find Lane Lines")

    images = sorted(os.listdir(img_dir))
    images = [f for f in images if f.endswith('.png')]
    for img_file in images:
        img_name = img_file.split('.')[0]
        img_path = img_dir + img_file
        img = cv2.imread(img_path)

        # Find fitting polynomials for lane lines in the image
        poly_fit_l, poly_fit_r, coords_lane_l, coords_lane_r, slide_wins = find_fitting_polynomials(img)
        coord_x_lane_l, coord_y_lane_l = coords_lane_l[0], coords_lane_l[1]
        coord_x_lane_r, coord_y_lane_r = coords_lane_r[0], coords_lane_r[1]

        print("Fitting polynomial (left)  for '{}': {:.6f}, {:.4f}, {:.1f}".format(
            img_file, poly_fit_l[0], poly_fit_l[1], poly_fit_l[2]))
        print("Fitting polynomial (right) for '{}': {:.6f}, {:.4f}, {:.1f}".format(
            img_file, poly_fit_r[0], poly_fit_r[1], poly_fit_r[2]))

        # Visualize processes
        if out_dir is not None:
            out_img = np.copy(img)

            # Plot fitting polynomials
            img_height = img.shape[0]
            y_val = np.linspace(0, img_height - 1, num=img_height)
            x_val_l = poly_fit_l[0] * (y_val**2) + poly_fit_l[1] * y_val + poly_fit_l[2]
            x_val_r = poly_fit_r[0] * (y_val**2) + poly_fit_r[1] * y_val + poly_fit_r[2]

            # Draw the lane line proximation window
            for slide_win in slide_wins:
                cv2.rectangle(out_img, slide_win[0], slide_win[1], (0, 255, 0), 2)

            # Color left and right lane lines in red and blue respectively
            out_img[coord_y_lane_l, coord_x_lane_l] = [255, 0, 0]
            out_img[coord_y_lane_r, coord_x_lane_r] = [0, 0, 255]

            plt.imshow(out_img)
            plt.plot(x_val_l, y_val, color='yellow')
            plt.plot(x_val_r, y_val, color='yellow')

            outfile = out_dir + img_name + '.png'
            print("Store the slide window lane detected image to {}".format(outfile))
            plt.savefig(outfile)
            plt.close()


#%% Step 5 - Measure lane curvature
def compute_curvature_poly2(A, B, y_eval):
    return ((1 + (2 * A * y_eval + B)**2)**1.5) / np.abs(2*A)

def get_curvature_in_meter(poly_fit, y_eval):
    # Meters per pixel in x and y direction
    mpp_x, mpp_y = 3.7 / 640, 30 / 720

    # Convert poly_fit in pixels into meters
    # For xp = A*(yp^2) + B*(yp) + C in pixels
    # xm = mx * xp and ym = my * yp => xp = xm / mx, yp = ym / my
    # xm / mx = A*((ym/my)^2) + B*(ym/my) + C
    # xm = (mx/(my^2))*A*(ym^2) + (mx/my)*B*ym + mx*C
    # where xp and yp are x and y coordinates in pixels, and xm and ym are x and y coordinates in meters
    poly_adjuster = np.array([mpp_x / (mpp_y ** 2), mpp_x / mpp_y, mpp_x])
    poly_fit_m = poly_fit * poly_adjuster
    y_eval_m = mpp_y * y_eval
    return compute_curvature_poly2(poly_fit_m[0], poly_fit_m[1], y_eval_m)

def get_vehicle_offset(poly_fit_l, poly_fit_r, y_eval, img_width):
    # Meters per pixel in x and y direction
    mpp_x, mpp_y = 3.7 / 640, 30 / 720

    # Find x coordinate of the car position by computing the mid point of left and right lanes
    x_lane_l = poly_fit_l[0]*(y_eval**2) + poly_fit_l[1]*y_eval + poly_fit_l[2]
    x_lane_r = poly_fit_r[0]*(y_eval**2) + poly_fit_r[1]*y_eval + poly_fit_r[2]
    x_lane_mid = (x_lane_l + x_lane_r) / 2

    # Positive offset means the car is right of the center (negative for left)
    x_car_pos = (img_width) / 2
    x_car_offset = x_car_pos - x_lane_mid
    x_car_offset_m = mpp_x * x_car_offset
    return x_car_offset_m

def perform_curvature_and_offset_measure(img_dir):
    print_section_header("Measure Curvature and Offset")

    images = sorted(os.listdir(img_dir))
    images = [f for f in images if f.endswith('.png')]
    for img_file in images:
        img_name = img_file.split('.')[0]
        img_path = img_dir + img_file
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[0:2]

        # Find fitting polynomials for lane lines in the image
        poly_fit_l, poly_fit_r, _, _, _ = find_fitting_polynomials(img)

        # Compute lane curvature in meters at the bottom of the image
        curv_l = get_curvature_in_meter(poly_fit_l, img_height)
        curv_r = get_curvature_in_meter(poly_fit_r, img_height)

        # Compute car position (x offset from the center in meter at the bottom of the image)
        offset = get_vehicle_offset(poly_fit_l, poly_fit_r, img_height, img_width)

        print("{} - Curvatures: ({:.1f} m, {:.1f} m), Offset: {:.3f} m".format(img_name, curv_l, curv_r, offset))


#%% Run lane detection on test images
def detect_lane_images(steps):
    print("\n** Running lane detection on test images **")

    # Step 0 - Analyze test image
    if (not steps) or (0 in steps):
        img_path = test_img_dir + 'test3.jpg'
        analyze_test_image(img_path=img_path)

    # Step 1 - Correct image distortion
    if (not steps) or (1 in steps):
        pickle_file = results_dir + 'camera_cal.p'
        correct_image_distortion(pickle_file)

    # Step 2 - Create a binary image with lane pixels
    if (not steps) or (2 in steps):
        create_lane_pixel_images()

    # Step 3 - Transform image perspective
    if (not steps) or (3 in steps):
        perform_perspective_transforms()

    # Step 4 - Find lane lines
    if (not steps) or (4 in steps):
        perform_lane_line_detection_images(perspective_trans_dir, slide_win_dir)

    # Step 5 - Measure lane curvature
    if (not steps) or (5 in steps):
        perform_curvature_and_offset_measure(perspective_trans_dir)


#%% Run lane detection on provided video
def detect_lane_video(steps):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Lane Lines')
    parser.add_argument('-i', '--image', dest='run_image', action='store_true',
                        help='Execute lane detection on test images')
    parser.add_argument('-v', '--video', dest='run_video', action='store_true',
                        help='Execute lane detection on test video')
    parser.add_argument('steps', metavar='step', type=int, nargs='*',
                        help='Provide steps to execute. Run all steps when omitted.')
    args  = parser.parse_args()
    print("Running lane detection with arguments: {}".format(args))

    if args.run_image:
        detect_lane_images(args.steps)
    if args.run_video:
        detect_lane_video(args.steps)
