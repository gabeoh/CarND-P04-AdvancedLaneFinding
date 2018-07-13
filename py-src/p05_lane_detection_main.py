#%% Initialization
import os
import pickle
import argparse
from moviepy.editor import VideoFileClip

from my_util import print_section_header, analyze_test_image
import p05_01_correct_distortion as dist_correct
import p05_02_create_binary_lane as bin_lane
import p05_03_perspective_transform as p_trans
import p05_04_identify_lane_line as id_ll
import p05_05_curvature_offset as curv_off
import p05_06_overlay_annotate as over_annot

project_root_dir = '../'
output_dir = project_root_dir + 'output_images/'
results_dir = project_root_dir + 'results/'
test_img_dir = project_root_dir + 'test_images/'
test_video_dir = project_root_dir + 'test_videos/'
undistorted_img_dir = output_dir + 'undistorted/'
binary_lane_dir = output_dir + 'binary_lanes/'
perspective_trans_dir = output_dir + 'perspective/'
slide_win_dir = output_dir + 'slide_win/'
overlay_dir = output_dir + 'overlay/'
video_dst_dir = output_dir + 'video/'


#%% Run lane detection on test images
def detect_lane_images(steps):
    print("\n** Running lane detection on test images **")

    pickle_file = results_dir + 'camera_cal.p'

    # Step 0 - Analyze test image
    if (not steps) or (0 in steps):
        print_section_header("Test Images")
        img_path = test_img_dir + 'test3.jpg'
        analyze_test_image(img_path=img_path)

    # Step 1 - Correct image distortion
    if (not steps) or (1 in steps):
        pickle_file = results_dir + 'camera_cal.p'
        dist_correct.perform_distortion_correction(pickle_file, test_img_dir, undistorted_img_dir)

    # Step 2 - Create a binary image with lane pixels
    if (not steps) or (2 in steps):
        bin_lane.perform_binary_lane_pixel_image_creation(undistorted_img_dir, binary_lane_dir)

    # Step 3 - Transform image perspective
    if (not steps) or (3 in steps):
        p_trans.perform_perspective_transforms(undistorted_img_dir, binary_lane_dir, perspective_trans_dir)

    # Step 4 - Find lane lines
    if (not steps) or (4 in steps):
        id_ll.perform_lane_line_detection_images(perspective_trans_dir, slide_win_dir)

    # Step 5 - Measure lane curvature
    if (not steps) or (5 in steps):
        curv_off.perform_curvature_and_offset_measure(perspective_trans_dir)

    # Step 6 - Overlay lane lines on original image
    if (not steps) or (6 in steps):
        over_annot.perform_lane_overlay_and_annotate(test_img_dir, pickle_file, overlay_dir)


#%% Run lane detection on provided video
def detect_lane_video(video_files):

    print("\n** Running lane detection on video files **")
    print('Video Files: ', video_files)

    # Load distortion correction parameters
    pickle_file = results_dir + 'camera_cal.p'
    with open(pickle_file, 'rb') as inf:
        camera_cal = pickle.load(inf)
    camera_mtx = camera_cal['camera_matrix']
    dist_coeffs = camera_cal['distortion_coefficients']

    # Get matrices for perspective transform and its reverse operation
    mtx_trans, mtx_trans_inv = p_trans.compute_perspective_transform_matrix()


    videos = sorted(os.listdir(test_video_dir))
    videos = [f for f in videos if f.endswith('.mp4') and (len(video_files) == 0 or f in video_files)]
    for video_file in videos:
        video_path = test_video_dir + video_file
        video_out_path = video_dst_dir + video_file
        print_section_header("Run lane detection on video - {}".format(video_file), 60)

        # Use subclip() to test with shorter video (the first 5 seconds for example)
        # clip = VideoFileClip(video_path).subclip(0,5)
        clip = VideoFileClip(video_path)
        clip_processed = clip.fl_image(lambda img: over_annot.overlay_lane_lines(img, camera_mtx, dist_coeffs, mtx_trans, mtx_trans_inv))

        # Write line detected videos to files
        clip_processed.write_videofile(video_out_path, audio=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Lane Lines')
    parser.add_argument('-i', '--image', dest='run_image', action='store_true',
                        help='Execute lane detection on test images')
    parser.add_argument('-v', '--video', dest='run_video', action='store_true',
                        help='Execute lane detection on test video')
    parser.add_argument('-f', '--files', dest='files', type=str, nargs='*',
                        help='Provide image/video file(s) to process. Run on all test files when omitted.')
    parser.add_argument('steps', metavar='step', type=int, nargs='*',
                        help='Provide steps to execute. Run all steps when omitted.')
    args  = parser.parse_args()
    print("Running lane detection with arguments: {}".format(args))

    files = [] if args.files is None else args.files
    if args.run_image:
        detect_lane_images(args.steps)
    if args.run_video:
        detect_lane_video(files)
