#%% Initialization
import numpy as np
import cv2
import os

from my_util import print_section_header, compute_curvature_poly2
import p05_04_identify_lane_line as id_ll


#%% Step 5 - Measure lane curvature
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_height, img_width = img.shape[0:2]

        # Find fitting polynomials for lane lines in the image
        poly_fit_l, poly_fit_r, _, _, _, _ = id_ll.find_fitting_polynomials(img)

        # Compute lane curvature in meters at the bottom of the image
        curv_l = get_curvature_in_meter(poly_fit_l, img_height)
        curv_r = get_curvature_in_meter(poly_fit_r, img_height)

        # Compute car position (x offset from the center in meter at the bottom of the image)
        offset = get_vehicle_offset(poly_fit_l, poly_fit_r, img_height, img_width)

        print("{} - Curvatures: ({:.1f} m, {:.1f} m), Offset: {:.3f} m".format(img_name, curv_l, curv_r, offset))

if __name__ == '__main__':
    # Step 5 - Measure lane curvature
    perspective_trans_dir = '../output_images/perspective/'
    perform_curvature_and_offset_measure(perspective_trans_dir)
