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

class DepthPublisher(Node):
    def __init__(self):
        super().__init__('depth_publisher')
        
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)

        # Get parameters
        self.frame_width = self.get_parameter('frame_width').get_parameter_value().integer_value
        self.frame_height = self.get_parameter('frame_height').get_parameter_value().integer_value

        # Load TRT engine
        self.engine = self.load_trt_engine('../../model_converter/model.engine')
        self.context = self.engine.create_execution_context()
        self.intrinsics = self.get_intrinsics('../../rosslam/camera_params/intrincities.xml')
        # Basic setup
        self.bridge = CvBridge()

        # Images subscription
        self.left_image_sub = message_filters.Subscriber(self, 'left_image', Image)
        self.right_image_sub = message_filters.Subscriber(self, 'right_image', Image)
        self.get_logger().info('Synchronizing images')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.left_image_sub, self.right_image_sub], 10, 0.1, allow_headless=True)
        self.ts.registerCallback(self.image_callback)
        # Disparity publisher
        self.depth_pub = self.create_publisher(Depth, 'depth_publisher', 10)
    
    def load_trt_engine(self, engine_path):
        '''
        Load TRT engine
        '''
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    
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

    def process_images(self, left_img, right_img):
        '''
        Convert images to CHW format
        Run inference
        Return disparity
        '''
        rectified_left = cv.remap(left_img, self.intrinsics['Left_Stereo_Map_x'], self.intrinsics['Left_Stereo_Map_y'], cv.INTER_LINEAR)
        rectified_right = cv.remap(right_img, self.intrinsics['Right_Stereo_Map_x'], self.intrinsics['Right_Stereo_Map_y'], cv.INTER_LINEAR)
        target_height, target_width = self.frame_height, self.frame_width
        left_img_res = cv.resize(rectified_left, (target_width, target_height))
        right_img_res = cv.resize(rectified_right, (target_width, target_height))

        # Convert to CHW format for TRT and ensure they are contiguous
        left_img_chw = left_img_res.transpose((2, 0, 1)).astype('float32')
        right_img_chw = right_img_res.transpose((2, 0, 1)).astype('float32')

        input_tensor = np.ascontiguousarray(np.array([left_img_chw, right_img_chw]))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        output = np.empty((1, target_height, target_width, 1), dtype=np.float32)
        d_input = cuda.mem_alloc(input_tensor.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)

        self.context.set_tensor_address(self.engine.get_tensor_name(0), int(d_input)) # input buffer
        self.context.set_tensor_address(self.engine.get_tensor_name(1), int(d_output)) # output buffer

        stream = cuda.Stream()
        # Copy images to the GPU
        cuda.memcpy_htod_async(d_input, input_tensor, stream)

        #  bindings = [int(d_input), int(d_output)]

        success = self.context.execute_async_v3(stream_handle=stream.handle)
        # self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        
        print(success)
        # Copy result from GPU
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
        output = np.squeeze(output, axis=(0, 3))  # Shape: [480, 640]

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
