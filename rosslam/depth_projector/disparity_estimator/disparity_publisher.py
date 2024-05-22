import rclpy
from rclpy.node import Node
import cv2 as cv
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
        engine = load_trt_engine('model.engine')

        context = engine.create_execution_context()

        
        target_height, target_width = 720, 1280 # Sizes for current model TODO params of the model
        left_img_res = cv.resize(left_img, (target_width, target_height))
        right_img_res = cv.resize(right_img, (target_width, target_height))

        # Convert to CHW format for TRT and ensure they are contiguous
        left_img_chw = left_img_res.transpose((2, 0, 1)).astype('float32')
        right_img_chw = right_img_res.transpose((2, 0, 1)).astype('float32')

        input_tensor = np.ascontiguousarray(np.array([left_img_chw, right_img_chw]))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        output = np.empty((1, 720, 1280, 1), dtype=np.float32)
        d_input = cuda.mem_alloc(input_tensor.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)

        context.set_tensor_address(engine.get_tensor_name(0), int(d_input)) # input buffer
        context.set_tensor_address(engine.get_tensor_name(1), int(d_output)) # output buffer

        stream = cuda.Stream()
        # Copy images to the GPU
        cuda.memcpy_htod_async(d_input, input_tensor, stream)

        success = context.execute_async_v3(stream_handle=stream.handle)
        print(success)
        # Copy result from GPU
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
        output = np.squeeze(output, axis=(0, 3))  # Shape: [720, 1280]

        return output

def main(args=None):
    rclpy.init(args=args)
    stereo_image_processor = DisparityPublisher()
    rclpy.spin(stereo_image_processor)
    stereo_image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
