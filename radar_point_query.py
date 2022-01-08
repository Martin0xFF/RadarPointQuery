'''
Visualize radar data by projecting the points into a provided image
'''

import numpy as np
import cv2

def colour_bgr(values, minimum=0, maximum=5):
    '''
    A Crude range to colour function:

    Values - some n length array of data

    Returns:
        - nx3 ndarray of data corresponding to the
        values of the input array
    '''
    #minimum, maximum = np.min(values), np.max(values)
    ratio = np.minimum(1, 2*(values-minimum)/(maximum - minimum))
    blu = np.maximum(0, 255*(1-ratio)).astype(int)
    red = np.maximum(0, 255*(ratio-1)).astype(int)
    gre = 255 - blu - red
    return np.vstack([blu, gre, red])

def weight_from_range(ranges, min_circle=1, max_circle=100):
    '''
    Convert an array of values to the sizes of a circle
        -ranges is of length n
    returns:
        - array of length n with values proportional to min_circle to
        max_circle
    '''
    ranges = 1/(ranges + 1)
    # minimum, maximum = np.min(ranges), np.max(ranges)
    minimum, maximum = 0, 1
    ratio = (ranges-minimum)/(maximum - minimum)
    return (max_circle - min_circle)*ratio + min_circle

def to_image_pixel(points, cam_int, tran):
    '''
    points - ndarray with shape [:3, :n], n is the number of points
        the rows follow x,y,z (i.e. row 0 represents all x coords of
        points). tranhe points are in the radar frame
    cam_int - camera intrinics matrix 3x3, can be determine in matlab using
        http://www.vision.caltech.edu/bouguetj/calib_doc/
    tran - extrinsics matrix 4x4, the transform from radar frame to camera frame
        commonly denoted as T_camera_radar
    '''
    k_aug = np.vstack([np.hstack([cam_int, np.zeros((3,1))]), [0,0,0,1]])
    points_aug = np.vstack([points, np.ones((1, points.shape[1]))])
    points_cam = tran@points_aug
    pixel_points = (k_aug@points_cam / points_cam[2,:])[:2, :]
    return pixel_points

def project_radarpoints_onto_img(img, pixel_points, radar_cross, velocity, radar_points):
    '''
    img - numpy nd array with the rows representing
        the height of the image and cols representing
        the width (r,c)
    radar_cross - column vector [:1,n] of the radar cross-sections,
        these should correspond to the points, i.e. point 0 has
        position x,y,z and radar_cross section rc
    velocity - [:2, :n], these velocities correspond to
        the points similarly to radar crossections. The first column
        is the velocity in the x direction and the second is the
        velocity in the y direction
    '''

    bgr = colour_bgr(np.sqrt(np.square(velocity[0, :]) +
                             np.square(velocity[1, :])))

    sizes = weight_from_range(np.sqrt(np.square(radar_points[0, :]) +
                                      np.square(radar_points[1, :]))).astype(int)

    for i, point in enumerate(pixel_points.T):
        cv2.circle(img, point.astype(int), sizes[i], bgr[:, i].tolist(), -1)

def within_box(bb, points):
    '''
    bb - bounding box defined as [xl, yl, xr, yr]
        - (xl,yl) position of top - left corner of box
        - (xr, yr) position of bottom - right corner of box
    points - ndarray with shape [:3, :n], n is the number of points
    '''
    in_box = (
        (points[0, :]>bb[0]) & (points[0, :]<bb[2]) &
        (points[1, :]>bb[1]) & (points[1, :]<bb[3])
    )

    return points[:, in_box]

if __name__ == "__main__":


    # Bounding Box test
    bb = [0, 0, 10, 10]
    pts = np.array([
        [1,100,2, 10 ],
        [1,5,5, 10 ],
        [69,100,200, 10],
    ])

    print(within_box(bb, pts))
