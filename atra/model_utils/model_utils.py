import peft
import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor

from atra.statics import MODEL_MAPPING
from atra.utils import timeit
from atra.model_utils.unlimiformer import Unlimiformer

last_request, last_response = None, None


@timeit
def get_model(model_class, model_id):
    model = model_class.from_pretrained(
        model_id,
        cache_dir="./model_cache",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    return model


@timeit
def get_processor(processor_class, model_id):
    processor = processor_class.from_pretrained(model_id)
    return processor


@timeit
def get_peft_model(peft_model_id, model_class) -> peft.PeftModel:
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

    return model


@timeit
def get_model_and_processor(lang: str, task: str, activate_cache=True):
    global last_request, last_response

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

    # construct the request string for the cache
    # the request string is a combination of model_id, task and config
    if adapter_id is not None:
        this_request = f"{adapter_id}"
    else:
        this_request = f"{model_id}"

    # check if the request is the same as the last request
    # if yes, return the last response (model, processor)
    if activate_cache and this_request == last_request:
        return last_response

    # load the model
    if adapter_id is not None:
        model = get_peft_model(adapter_id, model_class)
    else:
        model = get_model(model_class, model_id)

    # get processor
    processor_class = MODEL_MAPPING[task][base_model_lang].get(
        "processor", AutoProcessor
    )
    processor = get_processor(processor_class, model_id)

    # if Unlimiformer is available for the model type,
    # convert the model to a Unlimiformer model
    if model.config.model_type in ["bart", "led", "t5"]:
        model = Unlimiformer.convert_model(model).cuda()
        print("Unlimiformer model created")
    else:
        # convert the model to a BetterTransformer model
        try:
            model = BetterTransformer.transform(model)
        except Exception as e:
            print("Bettertransformer exception: ", e)

    # if the cache is activated and the last request is not None
    # delete the last response and empty the cache for the GPU
    if last_request is not None and activate_cache:
        try:
            last_response[0].cpu()
            del last_response
        except Exception as e:
            print("Error deleting last response: ", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # if the cache is activated, save the response as the last request and response
    if activate_cache:
        last_request = this_request
        last_response = model, processor

    return model, processor
