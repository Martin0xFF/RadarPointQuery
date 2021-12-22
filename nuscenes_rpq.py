import numpy as np
import cv2
import sys

from scipy.spatial.transform import Rotation as Rot
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

from radar_viz import project_radarpoints_onto_img

# Scipy uses quaterions of the form [x, y, z, w]
# Nuscenes returns quaterions of the form [w, x, y, z]
nusc = NuScenes(version='v1.0-mini', dataroot="data", verbose=True)
city_scene = nusc.scene[int(sys.argv[1])]
fs_token = city_scene['first_sample_token']
current_sample = nusc.get('sample', fs_token)

radar_type = "RADAR_FRONT"
cam_type = "CAM_FRONT"
no_filter = False
plot = False

if no_filter:
    RadarPointCloud.disable_filters()

write_dir = "imgs/"
counter = 1
while current_sample is not None:
    # Renderer examples below
    # nusc.render_sample(current_sample['token'])
    # nusc.render_sample_data(radar_front_data['token'], nsweeps=5, underlay_map=True)

    radar_sample_data = nusc.get('sample_data', current_sample['data'][radar_type])
    radar_cloud = RadarPointCloud.from_file("data/" + radar_sample_data['filename'])
    radar_points = radar_cloud.points
    radar_xyz = radar_points[0:3, :]

    # Radar extrinsics not that we do not need to invert rotation or pose
    radar_calibrated_sensor = nusc.get('calibrated_sensor', radar_sample_data['calibrated_sensor_token'])
    t_car_radar = np.array(radar_calibrated_sensor['translation'])
    q = radar_calibrated_sensor['rotation']
    q = q[1:] + q[0:1] # move w to the back
    C_car_radar = Rot.from_quat(q).as_matrix() # n.b. rotations are represented as quaterions
    T_car_radar = np.vstack([np.hstack([C_car_radar, t_car_radar[:,None]]), [[0,0,0,1]]])

    # Camera image data
    cam_sample_data = nusc.get('sample_data', current_sample['data'][cam_type])
    cam_img = cv2.imread("data/" + cam_sample_data['filename'])

    # Camera extrinsics matrix, note that we invert the rotation and pose
    cam_calibrated_sensor = nusc.get('calibrated_sensor', cam_sample_data['calibrated_sensor_token'])
    t_car_cam = np.array(cam_calibrated_sensor['translation'])
    q = cam_calibrated_sensor['rotation']
    q = q[1:] + q[0:1] # move w to the back
    C_car_cam = Rot.from_quat(q).as_matrix() # n.b. rotations are represented as quaterions

    t_cam_car = - C_car_cam.T@t_car_cam
    C_cam_car = C_car_cam.T

    T_cam_car = np.vstack([np.hstack([C_cam_car, t_cam_car[:,None]]), [[0,0,0,1]]])

    T_cam_radar = T_cam_car@T_car_radar
    K = np.array(cam_calibrated_sensor['camera_intrinsic'])

    project_radarpoints_onto_img(cam_img, radar_xyz, radar_points[5,:], radar_points[8:10, :], K, T_cam_radar)
    if plot:
        cv2.imshow(cam_type, cam_img)
        cv2.waitKey(0)

    cv2.imwrite(write_dir + f"{counter:04d}.png", cam_img)
    counter += 1
    if current_sample['next'] != '':
        current_sample = nusc.get('sample', current_sample['next'])
    else:
        current_sample = None
