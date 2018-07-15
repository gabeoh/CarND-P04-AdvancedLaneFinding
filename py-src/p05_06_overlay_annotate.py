#%% Initialization
import numpy as np
import cv2
import os
import pickle

from my_util import print_section_header
import p05_01_correct_distortion as dist_correct
import p05_02_create_binary_lane as bin_lane
import p05_03_perspective_transform as p_trans
import p05_04_identify_lane_line as id_ll
import p05_05_curvature_offset as curv_off


#%% Step 6 - Overlay lane lines on original image
def draw_lane_areas(img, img_height, poly_fit_l, poly_fit_r):
    y_val = np.linspace(0, img_height - 1, num=img_height)
    x_val_l = poly_fit_l[0] * (y_val ** 2) + poly_fit_l[1] * y_val + poly_fit_l[2]
    x_val_r = poly_fit_r[0] * (y_val ** 2) + poly_fit_r[1] * y_val + poly_fit_r[2]

    # Create an image to draw the lines on
    zeros_one_ch = np.zeros_like(img, dtype=np.uint8)
    img_lane_fill = np.dstack((zeros_one_ch, zeros_one_ch, zeros_one_ch))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_l = np.expand_dims(np.transpose(np.vstack([x_val_l, y_val])), axis=0)
    # Reverse (np.flipud()) the order of the right lane line to fill poly properly
    pts_r = np.expand_dims(np.flipud(np.transpose(np.vstack([x_val_r, y_val]))), axis=0)
    pts = np.hstack((pts_l, pts_r))

    # Draw the lane area filled with green color
    cv2.fillPoly(img_lane_fill, np.int_([pts]), (0, 255, 0))
    return img_lane_fill

def overlay_on_original_image(img, img_lane, mtx_trans_inv):
    img_height, img_width = img_lane.shape[0:2]
    # Transform the image back to the original perspective
    img_lane_trans_back = cv2.warpPerspective(img_lane, mtx_trans_inv, (img_width, img_height))
    # Overlay the transformed lane area on to the original image
    img_lane_overlay = cv2.addWeighted(img, 1, img_lane_trans_back, 0.3, 0)
    return img_lane_overlay

def overlay_lane_lines(img, camera_mtx, dist_coeffs, mtx_trans, mtx_trans_inv, prev_polys=None):
    img_height, img_width = img.shape[0:2]

    # 1) Undistort image
    img_undist = dist_correct.correct_image_distortion(img, camera_mtx, dist_coeffs)

    # 2) Create binary lane pixel image
    img_bin_lane = bin_lane.create_lane_pixel_images(img_undist)

    # 3) Transform image perspective to bird's eye view
    img_trans = p_trans.transform_image_perspective(img_bin_lane, mtx_trans)

    # 4) Find fitting polynomials for lane lines in the image
    poly_fit_l, poly_fit_r, coords_lane_l, coords_lane_r, lane_width_min_max, _ = id_ll.find_fitting_polynomials(img_trans, prev_polys)
    if (prev_polys is not None):
        prev_polys[0] = poly_fit_l
        prev_polys[1] = poly_fit_r

    # 5) Compute lane curvature in meters at the bottom of the image
    curv_l = curv_off.get_curvature_in_meter(poly_fit_l, img_height)
    curv_r = curv_off.get_curvature_in_meter(poly_fit_r, img_height)
    offset = curv_off.get_vehicle_offset(poly_fit_l, poly_fit_r, img_height, img_width)

    # 6a) Draw lane areas (+ color lane lines)
    img_lane_fill = draw_lane_areas(img_trans, img_height, poly_fit_l, poly_fit_r)
    img_lane_fill = id_ll.color_lane_lines(img_lane_fill, coords_lane_l, coords_lane_r)

    # 6b) Overlay lane area on to the original image (undistorted)
    img_overlay = overlay_on_original_image(img_undist, img_lane_fill, mtx_trans_inv)

    # 6c) Add curvature and offset text
    text_curv = "Radius of Curvature (left, right): {:.0f}m, {:.0f}m".format(curv_l, curv_r)
    text_offset = "Vehicle is {:.2f}m {} of center".format(np.abs(offset), 'right' if offset > 0 else 'left')
    cv2.putText(img_overlay, text_curv, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.putText(img_overlay, text_offset, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA)
    # Annotate lane width for debugging purpose
    if (False and lane_width_min_max is not None):
        text_lane_width = "Lane Width (Min, Max): {:.0f}, {:.0f} in pixels".format(lane_width_min_max[0], lane_width_min_max[1])
        cv2.putText(img_overlay, text_lane_width, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA)

    return img_overlay

def perform_lane_overlay_and_annotate(img_dir, pickle_file, out_dir):
    print_section_header("Overlay Lane Lines")

    # Load distortion correction parameters
    with open(pickle_file, 'rb') as inf:
        camera_cal = pickle.load(inf)
    camera_mtx = camera_cal['camera_matrix']
    dist_coeffs = camera_cal['distortion_coefficients']

    # Get matrices for perspective transform and its reverse operation
    mtx_trans, mtx_trans_inv = p_trans.compute_perspective_transform_matrix()

    images = sorted(os.listdir(img_dir))
    images = [f for f in images if f.endswith('.jpg')]
    for img_file in images:
        img_name = img_file.split('.')[0]
        img_path = img_dir + img_file
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_overlay = overlay_lane_lines(img, camera_mtx, dist_coeffs, mtx_trans, mtx_trans_inv)

        # Save output files
        outfile = out_dir + img_name + '.png'
        print("Store the lane area overlayed image to {}".format(outfile))
        cv2.imwrite(outfile, cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    # Step 6 - Overlay lane lines on original image
    pickle_file = '../results/camera_cal.p'
    test_img_dir = '../test_images/'
    overlay_dir = '../output_images/overlay/'
    perform_lane_overlay_and_annotate(test_img_dir, pickle_file, overlay_dir)
