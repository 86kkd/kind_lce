import tensorflow as tf 
from tensorflow.python.compiler.tensorrt import trt_convert as trt

saved_model_dir = './model/saved_model' # 请替换为你的模型路径

params = trt.DEFAULT_TRT_CONVERSION_PARAMS
params = params._replace(max_workspace_size_bytes=(1<<32)) # 修改workspace大小
params = params._replace(precision_mode="FP16") # 设置推理精度
params = params._replace(maximum_cached_engines=100) # 设置最大缓存引擎数量
print("\033[92mstart load model\033[0m")
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=saved_model_dir,
    conversion_params=params
)
print("\033[92mload model successfully\033[0m")
converter.convert()
converter.save("./model/saved_model_trt")
print("\033[92msave model success\033[0m")

