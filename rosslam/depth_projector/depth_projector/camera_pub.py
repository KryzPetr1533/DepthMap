import rclpy                     
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('camera_pub')
        self.declare_parameter('fps', 30)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)

        # Get parameters
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.frame_width = self.get_parameter('frame_width').get_parameter_value().integer_value
        self.frame_height = self.get_parameter('frame_height').get_parameter_value().integer_value

        self.get_logger().info('Camera fps: {}'.format(self.fps))
        self.get_logger().info('Camera frame width: {}'.format(self.frame_width))
        self.get_logger().info('Camera frame height: {}'.format(self.frame_height))

        self.left_image = self.create_publisher(Image, 'left_image', 10)
        self.right_image = self.create_publisher(Image, 'right_image', 10)
        
        
        self.cap_left = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap_right = cv2.VideoCapture(1, cv2.CAP_V4L2)

        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap_left.set(cv2.CAP_PROP_FPS, self.fps)

        self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap_right.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap_left.isOpened():
            self.get_logger().error('Failed to open left camera')
        if not self.cap_right.isOpened():
            self.get_logger().error('Failed to open right camera')

        self.cv_bridge = CvBridge()
        
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()

        if ret_left:
            self.left_image.publish(
                self.cv_bridge.cv2_to_imgmsg(frame_left, 'bgr8'))
        else:
            self.get_logger().info('No left frame')
        
        if ret_right:
            self.right_image.publish(
                self.cv_bridge.cv2_to_imgmsg(frame_right, 'bgr8'))
        else:
            self.get_logger().info('No right frame')

        self.get_logger().info('Publishing video frame')

    def destroy_node(self):
        # Release camera resources
        self.cap_left.release()
        self.cap_right.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()