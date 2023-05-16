import peft
import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor

from atra.statics import MODEL_MAPPING
from atra.utils import timeit
from atra.model_utils.unlimiformer import Unlimiformer

LAST_MODEL = None

MODELS_CACHE = {}


def get_model(model_class, model_id):
    global MODELS_CACHE
    model = MODELS_CACHE.get(model_id, None)
    cached = True
    if model is None:
        model = model_class.from_pretrained(
            model_id,
            cache_dir="./model_cache",
        )
        MODELS_CACHE[model_id] = model
        cached = False
    return model, cached


def get_processor(processor_class, model_id):
    processor = processor_class.from_pretrained(model_id)
    return processor


def get_peft_model(peft_model_id, model_class) -> peft.PeftModel:
    global MODELS_CACHE
    model = MODELS_CACHE.get(peft_model_id, None)
    cached = True
    if model is None:
        # Load the PEFT model
        peft_config = peft.PeftConfig.from_pretrained(peft_model_id)
        model = model_class.from_pretrained(
            peft_config.base_model_name_or_path,
            cache_dir="./model_cache",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model = peft.PeftModel.from_pretrained(
            model,
            peft_model_id,
            cache_dir="./model_cache",
        )
        model = model.merge_and_unload()
        model = model.eval()
        MODELS_CACHE[peft_model_id] = model
        cached = False

    return model, cached


@timeit
def get_model_and_processor(lang: str, task: str):
    global LAST_MODEL

    try:
        LAST_MODEL.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(e)

    # get all the model information from the mapping
    # for the requested task, config and lang
    model_id = MODEL_MAPPING[task].get(lang, {}).get("name", None)
    adapter_id = MODEL_MAPPING[task].get(lang, {}).get("adapter_id", None)
    if model_id is None:
        base_model_lang = "universal"
        model_id = MODEL_MAPPING[task][base_model_lang]["name"]
    else:
        base_model_lang = lang
    model_class = MODEL_MAPPING[task][base_model_lang].get("class", None)

    # load the model
    if adapter_id is not None:
        model, cached = get_peft_model(adapter_id, model_class)
    else:
        model, cached = get_model(model_class, model_id)

    # get processor
    processor_class = MODEL_MAPPING[task][base_model_lang].get(
        "processor", AutoProcessor
    )
    processor = get_processor(processor_class, model_id)

    if cached is False:
        # if Unlimiformer is available for the model type,
        # convert the model to a Unlimiformer model
        if model.config.model_type in ["bart", "led", "t5"]:
            model = Unlimiformer.convert_model(model)
            print("Unlimiformer model created")
        else:
            # convert the model to a BetterTransformer model
            try:
                model = BetterTransformer.transform(model)
            except Exception as e:
                print("Bettertransformer exception: ", e)

    if torch.cuda.is_available():
        model = model.cuda()

    LAST_MODEL = model

    return model, processor
