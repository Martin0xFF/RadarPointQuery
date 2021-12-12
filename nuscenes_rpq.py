import numpy as np
import cv2

from scipy.spatial.transform import Rotation as Rot
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

# Scipy uses quaterions of the form [x, y, z, w]
# Nuscenes returns quaterions of the form [w, x, y, z]

RadarPointCloud.disable_filters()

nusc = NuScenes(version='v1.0-mini', dataroot="data", verbose=True)
city_scene = nusc.scene[1]
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

    for point in P_img[0:2].T:
        cv2.circle(cam_image, tuple(point.astype(int)), 3, (255,0,0))
    cv2.imshow(cam_type, cam_image)
    cv2.waitKey(0)


    if current_sample['next'] != '':
        current_sample = nusc.get('sample', current_sample['next'])
    else:
        current_sample = None
