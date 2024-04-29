import os
import json
import errno
import numpy as np
import cv2
import pickle

def plot_point2d_on_img(img, point_cloud, color=(0, 255, 0), thickness=1):
    """Plot the converted 2D point cloud on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        point_cloud (numpy.array): Coordinates of the converted 2D Point Cloud.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """

    for i in range(len(point_cloud)):
        point = point_cloud[i].astype(np.int32)
        cv2.circle(img, point, 1, color, thickness)

    return img.astype(np.uint8)

def get_rgb(img_path):
    return cv2.imread(img_path)