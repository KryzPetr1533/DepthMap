import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
from msgs_definitions.msg import Depth
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

class DepthPublisher(Node):
    def __init__(self):
        super().__init__('depth_publisher')
        
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)

        # Get parameters
        self.frame_width = self.get_parameter('frame_width').get_parameter_value().integer_value
        self.frame_height = self.get_parameter('frame_height').get_parameter_value().integer_value

        # Load TRT engine
        self.engine = self.load_trt_engine('/var/model_converter/model.engine')
        if self.engine is None:
            self.get_logger().error("Failed to load TensorRT engine.")
            return

        self.execution_context = self.engine.create_execution_context()
        if self.execution_context is None:
            self.get_logger().error("Failed to create TensorRT execution context.")
            return
        self.intrinsics = self.get_intrinsics('/var/camera_params/intrinsics.xml')
        # Basic setup
        self.bridge = CvBridge()

        # Images subscription
        self.left_image_sub = message_filters.Subscriber(self, Image, 'left_image')
        self.right_image_sub = message_filters.Subscriber(self, Image, 'right_image')
        self.get_logger().info('Synchronizing images')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.left_image_sub, self.right_image_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)
        # Disparity publisher
        self.depth_pub = self.create_publisher(Depth, 'depth_publisher', 10)
    
    def load_trt_engine(self, engine_path):
        '''
        Load TRT engine
        '''
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        if not os.path.exists(engine_path):
            self.get_logger().error(f"Engine file not found at {engine_path}")
            return None
        
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            try:
                engine = runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    script_dir = os.path.dirname(os.path.realpath(__file__))
                    self.get_logger().error(f"Failed to load engine from {script_dir}")
                return engine
            except Exception as e:
                self.get_logger().error(f"Exception while loading engine: {e}")
                return None
    
    def get_intrinsics(self, file_path):
        # Open the file storage for reading
        cv_file = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)

        intrinsics = {}
        # Read the left stereo map (x coordinates)
        intrinsics['Left_Stereo_Map_x'] = cv_file.getNode("Left_Stereo_Map_x").mat()

        # Read the left stereo map (y coordinates)
        intrinsics['Left_Stereo_Map_y'] = cv_file.getNode("Left_Stereo_Map_y").mat()

        # Read the right stereo map (x coordinates)
        intrinsics['Right_Stereo_Map_x'] = cv_file.getNode("Right_Stereo_Map_x").mat()

        # Read the right stereo map (y coordinates)
        intrinsics['Right_Stereo_Map_y'] = cv_file.getNode("Right_Stereo_Map_y").mat()

        # Read the rectified camera matrix
        intrinsics['Rectifyed_mat_left'] = cv_file.getNode("Rectifyed_mat_left").mat()
        intrinsics['Mat_left'] = cv_file.getNode("Mat_left").mat()

        # Read the baseline (distance between the two cameras)
        intrinsics['Baseline'] = cv_file.getNode("Baseline").real()

        return intrinsics

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
        depth = self.disparity_to_depth(disparity_data)

        msg = Depth()
        msg.height = depth.shape[0]
        msg.width = depth.shape[1]
        msg.matr = depth.flatten().tolist()
        msg.intrinsics = self.intrinsics['Rectifyed_mat_left'].flatten().tolist()
        self.depth_pub.publish(msg)
        self.get_logger().info("Publishing depth.")

    def process_images(self, left_img, right_img):
        '''
        Convert images to CHW format
        Run inference
        Return disparity
        '''
        # self.get_logger().info("Frame width" + self.frame_width + ", frame height" + self.frame_height)
        # rectified_left = cv.remap(left_img, self.intrinsics['Left_Stereo_Map_x'], self.intrinsics['Left_Stereo_Map_y'], cv.INTER_LINEAR)
        # rectified_right = cv.remap(right_img, self.intrinsics['Right_Stereo_Map_x'], self.intrinsics['Right_Stereo_Map_y'], cv.INTER_LINEAR)
        
        # Turns out it's safer to take engine's params as base
        target_height, target_width = self.engine.get_tensor_shape(
            self.engine.get_tensor_name(0))[2], self.engine.get_tensor_shape(
                self.engine.get_tensor_name(0))[3]
        # left_img_res = cv.resize(rectified_left, (target_width, target_height))
        # right_img_res = cv.resize(rectified_right, (target_width, target_height))
        left_img_res = cv.resize(left_img, (target_width, target_height))
        right_img_res = cv.resize(right_img, (target_width, target_height))

        # Convert to CHW format for TRT and ensure they are contiguous
        left_img_chw = left_img_res.transpose((2, 0, 1)).astype('float32')
        right_img_chw = right_img_res.transpose((2, 0, 1)).astype('float32')

        input_tensor = np.ascontiguousarray(np.array([left_img_chw, right_img_chw]), dtype=np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        output = np.empty((1, target_height, target_width, 1), dtype=np.float32)
        d_input = cuda.mem_alloc(input_tensor.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)

        self.execution_context.set_tensor_address(self.engine.get_tensor_name(0), int(d_input)) # input buffer
        self.execution_context.set_tensor_address(self.engine.get_tensor_name(1), int(d_output)) # output buffer

        stream = cuda.Stream()
        # Copy images to the GPU
        cuda.memcpy_htod_async(d_input, input_tensor, stream)

        success = self.execution_context.execute_async_v3(stream_handle=stream.handle)
        
        if success:
            self.get_logger().info('Disparity is found.')
        else:
            self.get_logger().error('Engine cannot be executed.')
        
        # Copy result from GPU
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
        output = np.squeeze(output, axis=(0, 3))  # Shape: [480, 640]
        self.get_logger().info('Transfering disparity.')

        return output
    
    def disparity_to_depth(self, disparity):
        '''
        Convert disparity to depth
        '''
        # Filter some invalid values
        disparity[disparity <= 0] = None
        disparity[disparity > 200] = None
        return (self.intrinsics['Baseline'] * self.intrinsics['Mat_left'][0, 0]) / disparity
    
def main(args=None):
    rclpy.init(args=args)
    stereo_image_processor = DepthPublisher()
    rclpy.spin(stereo_image_processor)
    stereo_image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
