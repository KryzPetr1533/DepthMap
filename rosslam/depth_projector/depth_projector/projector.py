import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from msgs_definitions.msg import Depth


class DepthProjector(Node):

    def __init__(self):
        super().__init__('depth_projector')
        self.depth_io = self.create_subscription(
            Depth,
            'depth',
            self.depth_callback,
            10
        )
        self.pcl_io = self.create_publisher(
            PointCloud,
            'virtual_point_cloud',
            10
        )

    def depth_callback(self, msg):
        self.get_logger().info('Recieving depth message')
        h, w = msg.height, msg.width
        data = np.array(msg.matr).reshape(h, w)
        intr = np.array(msg.intrinsics).reshape(3, 3)
        pcl = self.depth_to_pcl(data, intr)

        msg = PointCloud()
        for point in pcl:
            x, y, z = point
            p = Point32()
            p.x, p.y, p.z = x, y, z
            msg.points.append(p)
        self.get_logger().info('Publishing pcl message')
        self.pcl_io.publish(msg)

    def depth_to_pcl(self, depth, intrinsics):
        i_dim, j_dim = depth.shape
        ij = np.mgrid[1: i_dim + 1: 1, 1: j_dim + 1: 1].reshape(2, -1).T.reshape(i_dim, j_dim, 2)
        z = np.ones((i_dim, j_dim, 1))
        ijz_matr = np.concatenate((ij, z), axis=-1)
        ijz_matr[:, :, 1] *= -1
        points = (ijz_matr * np.expand_dims(depth, -1)).reshape(-1, 3)
        points_in_cam_frame = np.transpose((np.linalg.inv(intrinsics) @ points.T), (1, 0))
        pointcloud = points_in_cam_frame[
            np.where(np.logical_not(np.all(points_in_cam_frame == 0.0, axis=1)))
        ]
        return pointcloud


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = DepthProjector()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()