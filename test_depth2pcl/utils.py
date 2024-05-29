import cv2 as cv
import numpy as np
import glob

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def depth_to_pcl(depth: np.array, intrinsics: np.array):
    """
    :arg depth: matrix with shape H x W
        with depth per pixel, 0 for sky and transparent
    :arg intrinsics: intrinsic matrix 3 x 3
    :return: pointcloud in 3D camera frame with shape N x 3
        where N is number of depths not equal 0 pixels
    """
    i_dim, j_dim = depth.shape
    ij = np.mgrid[1: i_dim + 1: 1, 1: j_dim + 1: 1].reshape(2, -1).T.reshape(i_dim, j_dim, 2)
    z = np.ones((i_dim, j_dim, 1))
    ijz_matr = np.concatenate((ij, z), axis=-1)
    ijz_matr[:,:,1] *= -1
    points = (ijz_matr * np.expand_dims(depth, -1)).reshape(-1, 3)
    points_in_cam_frame = np.transpose((np.linalg.inv(intrinsics) @ points.T), (1, 0))
    pointcloud = points_in_cam_frame[
        np.where(np.logical_not(np.all(points_in_cam_frame == 0.0, axis=1)))
    ]
    return pointcloud


def load_trt_engine(engine_path):
        '''
        Load TRT engine
        '''
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

def process_images(left_img, right_img):
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
        print(input_tensor.shape)
        print(engine.get_tensor_shape('input'))
        d_input = cuda.mem_alloc(input_tensor.nbytes)
        print(engine.get_tensor_shape('reference_output_disparity'))
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

def disparity_to_depth(disparity, baseline, f_pixel):
    return (baseline * f_pixel) / disparity

def calibration(path):
    # parameters
    chessboardSize = (6, 9)
    cellSize = 4.0  # size of the chessboard cell in cm

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare data structure
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []
    imgpointsL = []
    imgpointsR = []

    extention = ".jpg"
    imageL = "imgL"
    imageR = "imgR"
    imagesLeft = glob.glob(path + imageL + '*').sort()
    imagesRight = glob.glob(path + imageR + '*').sort()
    print(path + imageL + '*' + extention)

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):
        imgL = cv.imread(imgLeft)
        imgR = cv.imread(imgRight)
        outputL = imgL.copy()
        outputR = imgR.copy()
        grayL = cv.cvtColor(outputL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(outputR, cv.COLOR_BGR2GRAY)
        retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None,
                                                  cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None,
                                                  cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        if retR and retL:
            objpoints.append(objp)
            cv.cornerSubPix(grayR,
                            cornersR,
                            (2 * min(chessboardSize[0], chessboardSize[1]) - 1,
                             2 * min(chessboardSize[0], chessboardSize[1]) - 1),
                            (-1, -1),
                            criteria)
            cv.cornerSubPix(grayL,
                            cornersL,
                            (2 * min(chessboardSize[0], chessboardSize[1]) - 1,
                             2 * min(chessboardSize[0], chessboardSize[1]) - 1),
                            (-1, -1),
                            criteria)
            cv.drawChessboardCorners(outputR, chessboardSize, cornersR, retR)
            cv.drawChessboardCorners(outputL, chessboardSize, cornersL, retL)
            cv.imshow('corners', np.hstack((outputR, outputL)))
            cv.waitKey(0)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
    cv.destroyAllWindows()

    print("Calculating left camera parameters ... ")
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    hL, wL = grayL.shape[:2]
    new_mtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    print("Calculating right camera parameters ... ")
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
    hR, wR = grayR.shape[:2]
    new_mtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

    print("Stereo calibration .....")
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv.stereoCalibrate(objpoints,
                                                                                       imgpointsL,
                                                                                       imgpointsR,
                                                                                       new_mtxL,
                                                                                       distL,
                                                                                       new_mtxR,
                                                                                       distR,
                                                                                       grayL.shape[::-1],
                                                                                       criteria_stereo,
                                                                                       flags)
    
    rectify_scale = 1  # if 0 image croped, if 1 image not croped
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                                             grayL.shape[::-1], Rot, Trns,
                                                                             rectify_scale, (0, 0))

    Left_Stereo_Map_x, Left_Stereo_Map_y = cv.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                                 grayL.shape[::-1], cv.CV_16SC2)
    Right_Stereo_Map_x, Right_Stereo_Map_y = cv.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                                  grayR.shape[::-1], cv.CV_16SC2)
    print("Left_camera_matrix:", mtxL)
    print("Right_camera_matrix:", mtxR)
    print("Saving parameters ......")
    cv_file = cv.FileStorage(".\\dataparamspy.xml", cv.FILE_STORAGE_WRITE)
    cv_file.write("Left_cam_mat", mtxL)
    cv_file.write("Right_cam_mat", mtxR)
    # cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
    # cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
    # cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
    # cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
    cv_file.write("Trns", Trns)
    cv_file.release()
    print("Finishing Calibration ...")

    return Left_Stereo_Map_x, Left_Stereo_Map_y, Right_Stereo_Map_x, Right_Stereo_Map_y