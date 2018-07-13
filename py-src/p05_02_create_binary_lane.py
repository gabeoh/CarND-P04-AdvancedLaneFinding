#%% Initialization
import numpy as np
import cv2
import os

from my_util import print_section_header


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
def create_lane_pixel_images(img):
    bin_x_grad = get_binary_lane_pixels_x_grad(img)
    bin_grad_dir_mag = get_binary_lane_pixels_grad_dir_mag(img)
    bin_saturation = get_binary_lane_pixels_color_saturation(img)

    bin_combined = np.zeros_like(bin_x_grad)
    bin_combined[(bin_x_grad == 1) | (bin_grad_dir_mag == 1) | (bin_saturation == 1)] = 1
    img_bin_lane = bin_combined * 255
    return img_bin_lane

# Create a binary image with identified lane pixels using color and gradient thresholds
def perform_binary_lane_pixel_image_creation(img_dir, out_dir):
    print_section_header("Color Transform and Gradients")

    images = sorted(os.listdir(img_dir))
    images = [f for f in images if not f.startswith('.')]
    for img_file in images:
        img_name = img_file.split('.')[0]
        img_path = img_dir + img_file
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bin_lane = create_lane_pixel_images(img)

        # Save output files
        outfile = out_dir + img_name + '.png'
        print("Store the binary lane image to {}".format(outfile))
        cv2.imwrite(outfile, img_bin_lane)

if __name__ == '__main__':
    # Step 2 - Create a binary image with lane pixels
    undistorted_img_dir = '../output_images/undistorted/'
    binary_lane_dir = '../output_images/binary_lanes/'
    perform_binary_lane_pixel_image_creation(undistorted_img_dir, binary_lane_dir)
