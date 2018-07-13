#%% Initialization
import cv2
import os
import pickle

from my_util import print_section_header


#%% Step 1 - Correct image distortion
def correct_image_distortion(img, camera_mtx, dist_coeffs):
    img_undist = cv2.undistort(img, camera_mtx, dist_coeffs, None, camera_mtx)
    return img_undist

def perform_distortion_correction(pickle_file, img_dir, out_dir):
    print_section_header("Correct Image Distortions")

    # Load camera calibration parameters
    with open(pickle_file, 'rb') as inf:
        camera_cal = pickle.load(inf)
    camera_mtx = camera_cal['camera_matrix']
    dist_coeffs = camera_cal['distortion_coefficients']

    images = sorted(os.listdir(img_dir))
    images = [f for f in images if not f.startswith('.')]
    for img_file in images:
        # Read an image file and correct distortions
        img_name = img_file.split('.')[0]
        img_path = img_dir + img_file
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_undist = correct_image_distortion(img, camera_mtx, dist_coeffs)

        # Save output files
        outfile = out_dir + img_name + '.png'
        print("Store the undistorted image to {}".format(outfile))
        cv2.imwrite(outfile, cv2.cvtColor(img_undist, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # Step 1 - Correct image distortion
    pickle_file = '../results/camera_cal.p'
    test_img_dir = '../test_images/'
    undistorted_img_dir = '../output_images/undistorted/'
    perform_distortion_correction(pickle_file, test_img_dir, undistorted_img_dir)
