import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msg.msg import PointCloud2
import numpy as np

from rqp_tracker import RTracker

class RadarTracker(Node):

    def __init__(self, opt):
        super().__init__('radar_tracker')
        self.opt = opt  # cache the option

        self.tracker = RTracker()

        self.sub = self.create_subscription(PointCloud2, opt['radar_topic'],
            self.Update_Tracks, 0))

        # Write out the tracked points to files for now


    def decode_data(self, data):
        return np.array(data) 

    def callback_gen(self):
        sub_id = self.sub_num
        self.sub_num += 1

        def listener_callback(msg):
            for abox in msg.bbs:
                # found something we should log
                if abox.tracker_id in self.boxes_to_track[sub_id]:
                    # are we already logging it? if not initial it
                    if abox.tracker_id not in self.sub_data[sub_id]:
                        self.sub_data[sub_id][abox.tracker_id] = {el : [] for el in self.values_to_add}

                    values = self.decode_box(abox)
                    for value_type in self.sub_data[sub_id][abox.tracker_id]:
                        self.sub_data[sub_id][abox.tracker_id][value_type].append(values[value_type])

        return listener_callback

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot_boxes(self):
        for sub_id, boxes_dict in self.sub_data.items():
            for box_id, values_dict in boxes_dict.items():

                #if box_id not in self.plots[sub_id]:
                #    self.plots[sub_id][box_id] = dict()


