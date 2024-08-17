import tensorrt as trt
import os

def build_engine(model_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(model_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build the engine.")
        return None

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Engine successfully saved to {engine_path}")
    return engine_path

# Ensure the paths are correct
onnx_model_path = '/var/model_converter/model.onnx'
trt_engine_path = '/var/model_converter/model.engine'

if not os.path.exists(onnx_model_path):
    print(f"ONNX model file does not exist: {onnx_model_path}")
else:
    if build_engine(onnx_model_path, trt_engine_path) is None:
        print("Engine creation failed.")
