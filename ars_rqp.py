import numpy as np
import rclpy
import cv2
import point_cloud2

from numpy.linalg import inv

from scipy.spatial.transform import Rotation as R
from radar_viz import project_radarpoints_onto_img, to_image_pixel

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CompressedImage
from cv_bridge import CvBridge

# Magic numbers from https://gitlab.com/aUToronto/zeus/-/blob/develop/utils/tranforms/bolt_tf.launch
# doesnt account for lens distortion
r = R.from_euler('zyx', [-np.pi/2, 0 , 0])

print(r.as_matrix())

K = [
    [1101.218350, 0.000000, 958.260974],
    [0.000000, 1106.949096, 726.492834],
    [0.000000, 0.000000, 1.000000]
]


Tvm = np.array([
    [0, 1, 0, 0.64],
    [0, 0, 1, 0],
    [1, 0, 0, -0.45],
    [0,0,0,1]
])

Tir = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 2.73],
    [0, 0, 1, -1.215],
    [0,0,0,1]
])


Tiv = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0.45],
    [0,0,0,1]
])

Tmr = inv(Tvm)@inv(Tiv)@Tir
T_cam_radar = inv(Tmr)

class RadarPointQuerySub(Node):
    def __init__ (self):
        super().__init__('minimal_subscriber')
        self.radar_sub = self.create_subscription(
            PointCloud2,
            'radar_cartesian',
            self.listener_callback,
            10)

        self.img_sub = self.create_subscription(
            CompressedImage,
            'blackfly/image_color/compressed',
            self.img_callback,
            10)

        self.points = []
        self.cvbr = CvBridge()

    def listener_callback(self, msg):
        gen = point_cloud2.read_points(msg, ["x", "y", "z", "vel", "velvar", "rc0"])
        [self.points.append(p) for p in gen]

    def img_callback(self, img):
        if len(self.points) == 0:
            return
        cam_img = self.cvbr.compressed_imgmsg_to_cv2(img)
        radar_points = np.array(self.points).T
        self.points = []

        radar_xyz = radar_points[:3,:]
        pixel_points = to_image_pixel(
            radar_xyz,
            K,
            T_cam_radar,
        )
        project_radarpoints_onto_img(
            cam_img,
            pixel_points,
            radar_points[5, :],
            np.array([radar_points[3, :], radar_points[3, :]]),
            radar_xyz,
        )
        cam_img = cv2.resize(cam_img, (720, 480), interpolation=cv2.INTER_AREA)

        cv2.imshow("dank", cam_img)
        cv2.waitKey(10)


def main(args=None):
    rclpy.init(args=args)

    rqp_node = RadarPointQuerySub()
    rclpy.spin(rqp_node)
    rqp_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    print("starting projection")
    main()
