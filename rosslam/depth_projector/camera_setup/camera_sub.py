#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImageSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, 'stereo_image', 10)
        #self.sub = self.create_subscription(
        #    Image, 'image_raw', self.listener_callback, 10)
        self.sub_1 = message_filters.Subscriber(self, 'image_raw_1', Image)
        self.sub_2 = message_filters.Subscriber(self, 'image_raw_2', Image)
        message_filters.ApproximateTimeSynchronizer([subscription_1, subscription_2], 10, 1)

    def image_callback(self, msg1, msg2):
        img_1 = self.cv_bridge.imgmsg_to_cv2(msg1, 'bgra8')
        self.image_show(img_1)
        img_2 = self.cv_bridge.imgmsg_to_cv2(msg2, 'bgra8')
        self.image_show(img_2)
        self.publisher.publish(msg1)
        self.publisher.publish(msg2)

    def image_show(self, image):
        cv2.imshow("image", image)
        cv2.waitKey(10)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber("camera_sub")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
