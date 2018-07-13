import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def print_section_header(title, len_banner=35):
    """
    Helper function to print section header with given title
    :param title:
    :return:
    """
    print()
    print('#' * len_banner)
    print('#', title)
    print('#' * len_banner)

# Analyze image details
def analyze_test_image(img_path):
    img = cv2.imread(img_path)
    img_y_size, img_x_size = img.shape[0:2]
    print("Image File: {}".format(img_path))
    print("Image Size: {}x{}".format(img_x_size, img_y_size))
    print("Image Min/Max Values: ({}, {})".format(img.min(), img.max()))

def compute_curvature_poly2(A, B, y_eval):
    return ((1 + (2 * A * y_eval + B)**2)**1.5) / np.abs(2*A)

def capture_frame(video_path, t, out_file):
    clip = VideoFileClip(video_path)
    clip.save_frame(out_file, t)

# capture_frame('../test_videos/project_video.mp4', 39.6, '../test_images/additional1.jpg')
# capture_frame('../test_videos/project_video.mp4', 41.5, '../test_images/additional2.jpg')
# capture_frame('../test_videos/project_video.mp4', 41.6, '../test_images/additional3.jpg')
