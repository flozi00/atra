import onnxruntime
import os
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

cpu_threads = os.cpu_count()

providers = [
    #"CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
]

for p in providers:
    if p in onnxruntime.get_available_providers():
        provider = p
        break

provider_options = {}

if provider == "OpenVINOExecutionProvider":
    provider_options["num_of_threads": cpu_threads]

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)


def get_model_as_onnx(model_class, model_id):
    model = model_class.from_pretrained(
        model_id,
        provider=provider,
        session_options=sess_options,
        from_transformers=True,
        cache_dir="./model_cache",
    )
    save_directory = model_id.split("/")[-1] + "_onnx"

    model.save_pretrained(save_directory)

    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    try:
        quantizer = ORTQuantizer.from_pretrained(save_directory)
        quantizer.quantize(save_dir=save_directory,quantization_config=qconfig)
        model = model_class.from_pretrained(save_directory, file_name="model_quantized.onnx")
    except:
        encoder_quantizer = ORTQuantizer.from_pretrained(save_directory, file_name="encoder_model.onnx")
        decoder_quantizer = ORTQuantizer.from_pretrained(save_directory, file_name="decoder_model.onnx")
        decoder_wp_quantizer = ORTQuantizer.from_pretrained(save_directory, file_name="decoder_with_past_model.onnx")

        quantizer = [encoder_quantizer, decoder_quantizer, decoder_wp_quantizer]

        [q.quantize(save_dir=save_directory,quantization_config=qconfig) for q in quantizer]

        model = model_class.from_pretrained(save_directory, encoder_file_name="encoder_model_quantized.onnx", decoder_file_name="decoder_model_quantized.onnx", decoder_with_past_file_name="decoder_with_past_model_quantized.onnx")


    return model
