import numpy as np
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

RadarPointCloud.disable_filters()

nusc = NuScenes(version='v1.0-mini', dataroot="data", verbose=True)
city_scene = nusc.scene[0]
fs_token = city_scene['first_sample_token']
cs = nusc.get('sample', fs_token)
sensor_type = "RADAR_FRONT"

# Todo: extract image and use intrinscis to project front radar points into image plane of camera

while cs is not None:
    # nusc.render_sample(cs['token'])

    for key in cs['data']:
        if sensor_type in key:
            rpc = RadarPointCloud.from_file("data/" + nusc.get('sample_data', cs['data'][key])['filename'])
            print(rpc.points)
    # nusc.render_sample_data(radar_front_data['token'], nsweeps=5, underlay_map=True)
    cs = nusc.get('sample', cs['next'])

print(x)
