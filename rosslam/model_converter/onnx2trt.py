import tensorrt as trt

model_path = 'model.onnx'

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if not success:
    pass

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) #Тут надо посмотреть сколько точно надо
serialized_engine = builder.build_serialized_network(network, config)
print(type(serialized_engine))
with open('model.engine', 'wb') as f:
    f.write(serialized_engine)
