import numpy as np
import cv2

from scipy.spatial.transform import Rotation as Rot
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

def crude_bgr(values):
    minimum, maximum = np.min(values), np.max(values)
    ratio = 2*(values-minimum)/(maximum - minimum)
    b = np.maximum(0, 255*(1-ratio)).astype(int)
    r = np.maximum(0, 255*(ratio-1)).astype(int)
    g = 255 - b - r
    return np.vstack([b, g, r])


# Scipy uses quaterions of the form [x, y, z, w]
# Nuscenes returns quaterions of the form [w, x, y, z]

RadarPointCloud.disable_filters()

nusc = NuScenes(version='v1.0-mini', dataroot="data", verbose=True)
city_scene = nusc.scene[8]
fs_token = city_scene['first_sample_token']
current_sample = nusc.get('sample', fs_token)

radar_type = "RADAR_FRONT"
cam_type = "CAM_FRONT"

while current_sample is not None:
    # Renderer examples below
    # nusc.render_sample(current_sample['token'])
    # nusc.render_sample_data(radar_front_data['token'], nsweeps=5, underlay_map=True)

    radar_sample_data = nusc.get('sample_data', current_sample['data'][radar_type])
    radar_cloud = RadarPointCloud.from_file("data/" + radar_sample_data['filename'])
    radar_points = radar_cloud.points
    radar_xyz = radar_points[0:3, :]

    radar_calibrated_sensor = nusc.get('calibrated_sensor', radar_sample_data['calibrated_sensor_token'])
    t_car_radar = np.array(radar_calibrated_sensor['translation'])
    q = radar_calibrated_sensor['rotation']
    q = q[1:] + q[0:1] # move w to the back
    C_car_radar = Rot.from_quat(q).as_matrix() # n.b. rotations are represented as quaterions

    cam_sample_data = nusc.get('sample_data', current_sample['data'][cam_type])
    cam_image = cv2.imread("data/" + cam_sample_data['filename'])

    cam_calibrated_sensor = nusc.get('calibrated_sensor', cam_sample_data['calibrated_sensor_token'])
    t_car_cam = np.array(cam_calibrated_sensor['translation'])
    q = cam_calibrated_sensor['rotation']
    q = q[1:] + q[0:1] # move w to the back
    C_car_cam = Rot.from_quat(q).as_matrix() # n.b. rotations are represented as quaterions

    t_cam_car = - C_car_cam.T@t_car_cam
    C_cam_car = C_car_cam.T

    K = np.array(cam_calibrated_sensor['camera_intrinsic'])

    P_ego = t_car_radar[:, None] + C_car_radar@radar_xyz
    P_car = t_cam_car[:, None] + C_cam_car@P_ego
    P_img = K@P_car
    P_img = P_img/P_img[2,:]

    speeds = np.sqrt(np.square(radar_points[8, :]) + np.square(radar_points[9, :]))
    cross = radar_points[5, :]

    ranges = np.sqrt(np.square(radar_points[0, :]) + np.square(radar_points[1, :]))
    bgr = crude_bgr(speeds)

    def weight_from_range(ranges, min_circle=2, max_circle=20):
        # make closer circles are bigger
        ranges = 1/(ranges + 1)
        minimum, maximum = np.min(ranges), np.max(ranges)
        ratio = (ranges-minimum)/(maximum - minimum)
        return (max_circle - min_circle)*ratio + min_circle

    sizes = weight_from_range(ranges).astype(int)

    for index, point in enumerate(P_img[0:2].T):
        # 8 - vx_comp, 9 - vy_comp
        cv2.circle(cam_image, point.astype(int), sizes[index], bgr[:, index].tolist())

    cv2.imshow(cam_type, cam_image)
    cv2.waitKey(0)


    if current_sample['next'] != '':
        current_sample = nusc.get('sample', current_sample['next'])
    else:
        current_sample = None
