import tensorrt as trt
import os
import logging

def build_engine(model_path, engine_path, logger):
    logger_trt = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger_trt)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger_trt)

    logger.info("Reading and parsing the ONNX model from %s", model_path)
    with open(model_path, 'rb') as model_file:
        model_data = model_file.read()

        if not parser.parse(model_data):
            logger.error("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None

    logger.info("Setting builder configurations.")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    logger.info("Building the serialized engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Failed to build the engine.")
        return None

    logger.info(f"Saving the engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    logger.info(f"Engine successfully saved to {engine_path}")

# Ensure the paths are correct
onnx_model_paths = ['/var/model_converter/model.onnx', '/var/model_converter/model.ft.onnx']
trt_engine_paths = ['/rosslam/model_converter/model_720.engine', '/rosslam/model_converter/model_480.engine']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for onnx_model_path, trt_engine_path in zip(onnx_model_paths, trt_engine_paths):
    if not os.path.exists(onnx_model_path):
        logger.error(f"ONNX model file does not exist: {onnx_model_path}")
    else:
        if build_engine(onnx_model_path, trt_engine_path, logger) is None:
            logger.error("Engine creation failed.")
