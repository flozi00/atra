from optimum.bettertransformer import BetterTransformer

from aaas.utils import timeit
from aaas.statics import MODEL_MAPPING
from transformers import AutoProcessor
import torch


@timeit
def get_model(model_class, model_id):
    model = model_class.from_pretrained(
        model_id,
        cache_dir="./model_cache",
    )
    try:
        model = BetterTransformer.transform(model)
    except Exception as e:
        print(e)
    try:
        model = torch.compile(model)
    except Exception as e:
        print(e)
    return model


def get_model_and_processor(lang: str, task: str, config: str):

    model_id = MODEL_MAPPING[task][config].get(lang, {}).get("name", None)
    if model_id is None:
        lang = "universal"
        model_id = MODEL_MAPPING[task][config][lang]["name"]

    model_class = MODEL_MAPPING[task][config][lang].get("class", None)
    processor_class = MODEL_MAPPING[task][config][lang].get("processor", AutoProcessor)
    model = MODEL_MAPPING[task][config][lang].get("cachedmodel", None)
    processor = MODEL_MAPPING[task][config][lang].get("cachedprocessor", None)

    if model is None or processor is None:
        model = get_model(model_class, model_id)
        processor = processor_class.from_pretrained(model_id)
        MODEL_MAPPING[task][config][lang]["cachedmodel"] = model
        MODEL_MAPPING[task][config][lang]["cachedprocessor"] = processor

    return model, processor
