'''
Visualize radar data by projecting the points into a provided image
'''

import numpy as np
import cv2

def colour_bgr(values):
    '''
    A Crude range to colour function:

    Values - some n length array of data

    Returns:
        - nx3 ndarray of data corresponding to the
        values of the input array
    '''
    minimum, maximum = np.min(values), np.max(values)
    ratio = 2*(values-minimum)/(maximum - minimum)
    blu = np.maximum(0, 255*(1-ratio)).astype(int)
    red = np.maximum(0, 255*(ratio-1)).astype(int)
    gre = 255 - blu - red
    return np.vstack([blu, gre, red])

def weight_from_range(ranges, min_circle=2, max_circle=20):
    '''
    Convert an array of values to the sizes of a circle
        -ranges is of length n
    returns:
        - array of length n with values proportional to min_circle to
        max_circle
    '''
    ranges = 1/(ranges + 1)
    minimum, maximum = np.min(ranges), np.max(ranges)
    ratio = (ranges-minimum)/(maximum - minimum)
    return (max_circle - min_circle)*ratio + min_circle


def project_radarpoints_onto_img(img, points, radar_cross, velocity, K, T):
    '''
    img - numpy nd array with the rows representing
        the height of the image and cols representing
        the width (r,c)
    points - ndarray with shape [:3, :n], n is the number of points
        the rows follow x,y,z (i.e. row 0 represents all x coords of
        points). The points are in the radar frame
    radar_cross - column vector [:1,n] of the radar cross-sections,
        these should correspond to the points, i.e. point 0 has
        position x,y,z and radar_cross section rc
    velocity - [:2, :n], these velocities correspond to
        the points similarly to radar crossections. The first column
        is the velocity in the x direction and the second is the
        velocity in the y direction
    K - camera intrinics matrix 3x3, can be determine in matlab using
        http://www.vision.caltech.edu/bouguetj/calib_doc/
    T - extrinsics matrix 4x4, the transform from radar frame to camera frame
        commonly denoted as T_camera_radar
    '''

    k_aug = np.vstack([np.hstack([K, np.zeros((3,1))]), [0,0,0,1]])
    points_aug = np.vstack([points, np.ones((1, points.shape[1]))])
    points_cam = T@points_aug
    pixel_points = (k_aug@points_cam / points_cam[2,:])[:2, :]


    bgr = colour_bgr(np.sqrt(np.square(velocity[0, :]) +
                             np.square(velocity[1, :])))

    sizes = weight_from_range(np.sqrt(np.square(points[0, :]) +
                                      np.square(points[1, :]))).astype(int)
    for i, point in enumerate(pixel_points.T):
        cv2.circle(img, point.astype(int), sizes[i], bgr[:, i].tolist())
