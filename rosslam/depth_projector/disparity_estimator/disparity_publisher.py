import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from msgs_definitions.msg import DisparityData
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class DisparityPublisher(Node):
    def __init__(self):
        super().__init__('stereo_image_processor')
        # Load TRT engine
        self.engine = self.load_trt_engine('some_trt_engine_path')
        self.context = self.engine.create_execution_context()

        # Basic setup
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.5, self.timer_callback)  # Adjust timer as needed

        # Images subscription
        self.left_image_sub = message_filters.Subscriber(self, 'left_image', Image)
        self.right_image_sub = message_filters.Subscriber(self, 'right_image', Image)
        self.get_logger().info('Synchronizing images')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.left_image_sub, self.right_image_sub], 10, 0.1, allow_headless=True)
        self.ts.registerCallback(self.image_callback)
        # Disparity publisher
        self.disparity_pub = self.create_publisher(DisparityData, 'disparity_data', 10)
    
    def load_trt_engine(self, engine_path):
        '''
        Load TRT engine
        '''
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    
    def image_callback(self, left_msg, right_msg):
        '''
        Convert messages to images RGB (important)
        Run disparity function
        Publish disparity
        '''
        try:
            left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='rgb8') # Important to use rgb8 for TRT
            right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            self.get_logger().info('Failed to convert image: {}'.format(e))
            return
        
        disparity_data = self.process_images(left_img, right_img)
        msg = DisparityData()
        msg.height = disparity_data.shape[0]
        msg.width = disparity_data.shape[1]
        msg.matr = disparity_data.flatten().tolist()
        self.disparity_pub.publish(msg)

    def process_images(self, left_img, right_img):
        '''
        Convert images to CHW format
        Run inference
        Return disparity
        '''
        # Convert to CHW format for TRT
        left_img_chw = left_img.transpose((2, 0, 1))
        right_img_chw = right_img.transpose((2, 0, 1))

        d_left_img = cuda.mem_alloc(left_img_chw.nbytes)
        d_right_img = cuda.mem_alloc(right_img_chw.nbytes)
        d_disparity = cuda.mem_alloc(left_img_chw.nbytes)
        bindings = [int(d_left_img), int(d_right_img), int(d_disparity)]

        stream = cuda.Stream()
        # Copy images to the GPU
        cuda.memcpy_htod_async(d_left_img, left_img_chw, stream)
        cuda.memcpy_htod_async(d_right_img, right_img_chw, stream)

        # Run inference
        self.context.execute_async(bindings=bindings, stream_handle=stream.handle)

        # Copy result from GPU
        output = np.empty((right_img_chw.shape[1], right_img_chw.shape[2]), dtype=np.int16)
        cuda.memcpy_dtoh_async(output, d_disparity, stream)
        stream.synchronize()

        return output

def main(args=None):
    rclpy.init(args=args)
    stereo_image_processor = DisparityPublisher()
    rclpy.spin(stereo_image_processor)
    stereo_image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
