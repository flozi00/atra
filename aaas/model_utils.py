import onnxruntime

providers = [
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
]

for p in providers:
    if p in onnxruntime.get_available_providers():
        provider = p
        break

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

    return model
