import os

import onnxruntime
from optimum.bettertransformer import BetterTransformer

from aaas.utils import timeit
from aaas.statics import MODEL_MAPPING
from transformers import AutoProcessor


cpu_threads = os.cpu_count()

providers = [
    # "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
]

for p in providers:
    if p in onnxruntime.get_available_providers():
        provider = p
        break

provider_options = {}

if provider == "OpenVINOExecutionProvider":
    provider_options["num_of_threads":cpu_threads]

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)


@timeit
def get_model(model_class, model_id):
    try:
        model = model_class.from_pretrained(
            model_id,
            provider=provider,
            session_options=sess_options,
            from_transformers=True,
            cache_dir="./model_cache",
        )
        model.save_pretrained("./model_cache" + model_id.split("/")[-1])
    except Exception:
        model = model_class.from_pretrained(model_id, cache_dir="./model_cache",)
        try:
            model = BetterTransformer.transform(model)
        except Exception:
            pass
    return model


def get_model_and_processor(lang: str, task: str, config: str):

    model_id = MODEL_MAPPING[task][config].get(lang, {}).get("name", None)
    if model_id is None:
        lang = "universal"
        model_id = MODEL_MAPPING[task][config][lang]["name"]

    model_class = MODEL_MAPPING[task][config][lang].get("class", None)
    processor_class = MODEL_MAPPING[task][config][lang].get("processor", AutoProcessor)
    model = MODEL_MAPPING[task][config][lang].get("cachedmodel", None)

    processor = processor_class.from_pretrained(model_id)

    if model is None:
        model = get_model(model_class, model_id)
        MODEL_MAPPING[task][config][lang]["cachedmodel"] = model

    return model, processor
