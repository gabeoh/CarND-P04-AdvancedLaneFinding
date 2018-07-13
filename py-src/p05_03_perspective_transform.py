#%% Initialization
import numpy as np
import cv2
import os

from my_util import print_section_header


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

def transform_image_perspective(img, mtx_trans):
    img_height, img_width = img.shape[0:2]
    img_trans = cv2.warpPerspective(img, mtx_trans, (img_width, img_height), flags=cv2.INTER_LINEAR)
    return img_trans

# Demonstrate image perspective transform
def demonstrate_perspective_transform(img_dir, img_file, mtx_trans, output_dir=None, mark_guideline=False):
    img_path = img_dir + img_file
    img_name = img_file.split('.')[0]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform image perspective
    img_trans = transform_image_perspective(img, mtx_trans)

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

def perform_perspective_transforms(undistorted_img_dir, binary_lane_dir, perspective_trans_dir):
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

if __name__ == '__main__':
    # Step 3 - Transform image perspective
    undistorted_img_dir = '../output_images/undistorted/'
    binary_lane_dir = '../output_images/binary_lanes/'
    perspective_trans_dir = '../output_images/perspective/'
    perform_perspective_transforms(undistorted_img_dir, binary_lane_dir, perspective_trans_dir)
