from optimum.bettertransformer import BetterTransformer

from aaas.utils import timeit
from aaas.statics import MODEL_MAPPING
from transformers import AutoProcessor
import torch
from peft import PeftModel
from functools import cache


@cache
@timeit
def get_model(model_class, model_id):
    model = model_class.from_pretrained(
        model_id,
        cache_dir="./model_cache",
        device_map="auto",
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


@cache
@timeit
def get_processor(processor_class, model_id):
    processor = processor_class.from_pretrained(model_id)
    return processor


@cache
@timeit
def get_peft_model(model, peft_model_id):
    model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")
    return model


@cache
@timeit
def get_model_and_processor(lang: str, task: str, config: str):
    # get model id
    model_id = MODEL_MAPPING[task][config].get(lang, {}).get("name", None)
    # set universal model if no specific model is available
    if model_id is None:
        base_model_lang = "universal"
        model_id = MODEL_MAPPING[task][config][base_model_lang]["name"]

    # get model
    model_class = MODEL_MAPPING[task][config][base_model_lang].get("class", None)
    model = get_model(model_class, model_id)

    # get processor
    processor_class = MODEL_MAPPING[task][config][base_model_lang].get(
        "processor", AutoProcessor
    )
    processor = get_processor(processor_class, model_id)

    adapter_id = MODEL_MAPPING[task][config].get(lang, {}).get("adapter_id", None)
    if adapter_id is not None:
        model = get_peft_model(model, adapter_id)

    return model, processor
