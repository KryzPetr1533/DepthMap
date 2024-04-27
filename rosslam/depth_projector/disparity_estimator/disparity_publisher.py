import rclpy
from rclpy.node import Node
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os

from your_package_name.msg import DisparityData  # Import your custom message

class DisparityPublisher(Node):
    def __init__(self):
        super().__init__('stereo_image_processor')
        self.publisher_ = self.create_publisher(DisparityData, 'disparity_data', 10)
        self.bridge = CvBridge()
        self.left_images_dir = '/path/to/left/images'
        self.right_images_dir = '/path/to/right/images'
        self.timer = self.create_timer(0.5, self.timer_callback)  # Adjust timer as needed
        self.left_image_files = sorted(os.listdir(self.left_images_dir))
        self.right_image_files = sorted(os.listdir(self.right_images_dir))
        self.index = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model().to(self.device)
        self.model.eval()

    def load_model(self):
        model_path = 'path_to_your_model.pth'
        model = torch.load(model_path, map_location=self.device)
        return model

    def timer_callback(self):
        if self.index >= len(self.left_image_files) or self.index >= len(self.right_image_files):
            self.index = 0  # Reset or stop
            return

        left_img_path = os.path.join(self.left_images_dir, self.left_image_files[self.index])
        right_img_path = os.path.join(self.right_images_dir, self.right_image_files[self.index])

        left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

        disparity_data = self.process_images(left_img, right_img)

        msg = DisparityData()
        msg.height = disparity_data.shape[0]
        msg.width = disparity_data.shape[1]
        msg.data = disparity_data.flatten().tolist()
        self.publisher_.publish(msg)
        self.index += 1

    def process_images(self, left_img, right_img):
        '''
        '''
        left_tensor = torch.tensor(left_img).unsqueeze(0).unsqueeze(0).float().to(self.device)
        right_tensor = torch.tensor(right_img).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            disparity = self.model(left_tensor, right_tensor).squeeze().cpu().numpy()
        return disparity

def main(args=None):
    rclpy.init(args=args)
    stereo_image_processor = DisparityPublisher()
    rclpy.spin(stereo_image_processor)
    stereo_image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
