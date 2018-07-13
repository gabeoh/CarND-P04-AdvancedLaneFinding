#%% Initialization
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from my_util import print_section_header


#%% Step 4 - Identify lane line pixels
# Find lane line pixel coordinate indices through sliding windows
def find_lane_pixel_coordinate_indices(img_height, pix_coord_x, pix_coord_y, x_base_l, x_base_r):
    # Number of sliding windows
    n_windows = 9
    # Sliding window width: +/- wind_margin
    win_margin = 100
    # Minimum number of pixels to re-center sliding window
    th_recenter_min_pix = 100

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
    img_bin = np.zeros_like(img)
    img_bin[img > img.max() // 2] = 1
    img_height, img_width = img_bin.shape

    # Find x and y coordinates of nonzero pixels
    pix_coord_y, pix_coord_x = img_bin.nonzero()

    # Find x coordinate bases for lines using histogram of the bottom half of the image
    mid_x, mid_y = img_width // 2, img_height // 2
    hist_x = np.sum(img_bin[mid_y:, :], axis=0)
    # Disregard 1/8th of each side when determining the base x coordinates
    offset_disregard = img_width // 8
    x_base_l = np.argmax(hist_x[offset_disregard:mid_x - offset_disregard]) + offset_disregard
    x_base_r = np.argmax(hist_x[mid_x + offset_disregard:-offset_disregard]) + mid_x + offset_disregard

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

def color_lane_lines(img, coords_lane_l, coords_lane_r):
    out_img = np.copy(img)
    coord_x_lane_l, coord_y_lane_l = coords_lane_l[0], coords_lane_l[1]
    coord_x_lane_r, coord_y_lane_r = coords_lane_r[0], coords_lane_r[1]
    # Color left and right lane lines in red and blue respectively
    out_img[coord_y_lane_l, coord_x_lane_l] = [255, 0, 0]
    out_img[coord_y_lane_r, coord_x_lane_r] = [0, 0, 255]
    return out_img

def perform_lane_line_detection_images(img_dir, out_dir=None):
    print_section_header("Find Lane Lines")

    images = sorted(os.listdir(img_dir))
    images = [f for f in images if f.endswith('.png')]
    for img_file in images:
        img_name = img_file.split('.')[0]
        img_path = img_dir + img_file
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find fitting polynomials for lane lines in the image
        poly_fit_l, poly_fit_r, coords_lane_l, coords_lane_r, slide_wins = find_fitting_polynomials(img_gray)

        print("Fitting polynomial (left)  for '{}': {:.6f}, {:.4f}, {:.1f}".format(
            img_file, poly_fit_l[0], poly_fit_l[1], poly_fit_l[2]))
        print("Fitting polynomial (right) for '{}': {:.6f}, {:.4f}, {:.1f}".format(
            img_file, poly_fit_r[0], poly_fit_r[1], poly_fit_r[2]))

        # Visualize processes
        if out_dir is not None:
            out_img = np.copy(img)

            # Plot fitting polynomials
            img_height = out_img.shape[0]
            y_val = np.linspace(0, img_height - 1, num=img_height)
            x_val_l = poly_fit_l[0] * (y_val**2) + poly_fit_l[1] * y_val + poly_fit_l[2]
            x_val_r = poly_fit_r[0] * (y_val**2) + poly_fit_r[1] * y_val + poly_fit_r[2]

            # Draw the lane line proximation window
            for slide_win in slide_wins:
                cv2.rectangle(out_img, slide_win[0], slide_win[1], (0, 255, 0), 2)

            # Color left and right lane lines in red and blue respectively
            out_img = color_lane_lines(out_img, coords_lane_l, coords_lane_r)

            plt.imshow(out_img)
            plt.plot(x_val_l, y_val, color='yellow')
            plt.plot(x_val_r, y_val, color='yellow')

            outfile = out_dir + img_name + '.png'
            print("Store the slide window lane detected image to {}".format(outfile))
            plt.savefig(outfile)
            plt.close()

if __name__ == '__main__':
    # Step 4 - Find lane lines
    perspective_trans_dir = '../output_images/perspective/'
    slide_win_dir = '../output_images/slide_win/'
    perform_lane_line_detection_images(perspective_trans_dir, slide_win_dir)
