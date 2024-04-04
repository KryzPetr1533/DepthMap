import rclpy
from rclpy.node import Node

import sys
import numpy as np
from std_msgs.msg import String
from msgs_definitions.msg import Depth
import cv2


import os
names = list(map(
    lambda x: x.rstrip(".jpg"),
    os.listdir("/rosslam/test_depth_data/left_small/")
))
intrinsics = "/rosslam/test_depth_data/intrinsics.txt"


class MockDepthPublisher(Node):

    def read_intrinsics(self, file_path):
        with open(file_path) as fr:
            lines = fr.read().split('\n')

        intr = None
        for line in lines:
            if line.startswith('K_101: '):
                line = line.lstrip('K_101: ')
                intr = np.array(list(map(float, line.split()))).reshape(3, 3)
        return intr

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Depth, 'depth', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.intr = self.read_intrinsics(intrinsics)
        self.i = 0

    def timer_callback(self):
        if self.i >= len(names):
            return
        idx = self.i % len(names)
        fname = f'/rosslam/test_depth_data/depth_small/{names[idx]}.png'
        img = cv2.imread(fname)[:,:,0]
        msg = Depth()
        msg.matr = (img.reshape(-1) / 256).tolist()
        msg.intrinsics = self.intr.reshape(-1).tolist()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing %i-th depth message' % (self.i + 1,))
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MockDepthPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()