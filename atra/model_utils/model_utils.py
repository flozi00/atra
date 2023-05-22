import peft
import torch
from optimum.bettertransformer import BetterTransformer
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoProcessor
from atra.model_utils.unlimiformer import Unlimiformer

from atra.statics import MODEL_MAPPING
from atra.utils import timeit
import GPUtil
import gradio as gr

MODELS_CACHE = {}


def free_gpu():
    global MODELS_CACHE
    gpus = GPUtil.getGPUs()
    for gpu_num in range(len(gpus)):
        gpu: GPUtil.GPU = gpus[gpu_num]

        if gpu.memoryUtil * 100 > 60:
            models_list = list(MODELS_CACHE.keys())
            for model in models_list:
                if MODELS_CACHE[model]["on_gpu"] is True:
                    MODELS_CACHE[model]["model"].to("cpu")
                    MODELS_CACHE[model]["on_gpu"] = False
                    print("Model {} moved to CPU".format(model))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model(model_class, model_id) -> bool:
    """This function loads a model from cache or from the Huggingface model hub

    Args:
        model_class (PretrainedModel): The Huggingface model class
        model_id (str): The Huggingface model id to load

    Returns:
        PretrainedModel, bool: The loaded model and
          a boolean indicating if the model was loaded from cache
    """
    global MODELS_CACHE
    cached = MODELS_CACHE.get(model_id, None)
    if cached is None:
        model = model_class.from_pretrained(
            model_id,
            cache_dir="./model_cache",
        )
        MODELS_CACHE[model_id] = {"model": model, "on_gpu": False}
        return False
    else:
        return True


@timeit
def get_processor(processor_class, model_id):
    processor = processor_class.from_pretrained(model_id)
    return processor


def get_peft_model(peft_model_id, model_class) -> bool:
    """This function loads a model from cache or from the Huggingface model hub

    Args:
        model_class (PretrainedModel): The Huggingface model class
        model_id (str): The Huggingface model id to load

    Returns:
        PretrainedModel, bool: The loaded model and a boolean indicating if the model was loaded from cache
    """
    global MODELS_CACHE
    cached = MODELS_CACHE.get(peft_model_id, None)
    if cached is None:
        # Load the PEFT model
        peft_config = peft.PeftConfig.from_pretrained(peft_model_id)
        model = model_class.from_pretrained(
            peft_config.base_model_name_or_path,
            cache_dir="./model_cache",
        )
        model = peft.PeftModel.from_pretrained(
            model,
            peft_model_id,
            cache_dir="./model_cache",
        )
        model = model.merge_and_unload()  # type: ignore
        MODELS_CACHE[peft_model_id] = {"model": model, "on_gpu": False}
        return False
    else:
        return True


@timeit
def get_model_and_processor(
    lang: str, task: str, progress=gr.Progress()
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    global MODELS_CACHE

    # get all the model information from the mapping
    # for the requested task, config and lang
    progress.__call__(0.25, "Loading Model Information")
    model_id = MODEL_MAPPING[task].get(lang, {}).get("name", None)
    adapter_id = MODEL_MAPPING[task].get(lang, {}).get("adapter_id", None)
    if model_id is None:
        base_model_lang = "universal"
        model_id = MODEL_MAPPING[task][base_model_lang]["name"]
    else:
        base_model_lang = lang
    model_class = MODEL_MAPPING[task][base_model_lang].get("class", None)

    progress.__call__(0.3, "Loading Model tensors")
    # load the model
    if adapter_id is not None:
        cached = get_peft_model(adapter_id, model_class)
        id_to_use = adapter_id
    else:
        cached = get_model(model_class, model_id)
        id_to_use = model_id

    # get processor
    processor_class = MODEL_MAPPING[task][base_model_lang].get(
        "processor", AutoProcessor
    )

    progress.__call__(0.5, "Optimizing Model")
    if cached is False:
        if MODELS_CACHE[id_to_use]["model"].config.model_type == "t5":
            print("Converting T5 model to Unlimiformer")
            MODELS_CACHE[id_to_use]["model"] = Unlimiformer.convert_model(
                MODELS_CACHE[id_to_use]["model"]
            )
            MODELS_CACHE[id_to_use]["model"].eval()
        else:
            try:
                MODELS_CACHE[id_to_use]["model"] = BetterTransformer.transform(MODELS_CACHE[id_to_use]["model"])  # type: ignore
            except Exception as e:
                print("Bettertransformer exception: ", e)
            MODELS_CACHE[id_to_use]["model"] = torch.compile(
                MODELS_CACHE[id_to_use]["model"], mode="max-autotune", backend="onnxrt"
            )
        processor = get_processor(processor_class, model_id)
        MODELS_CACHE[id_to_use]["processor"] = processor

    progress.__call__(0.6, "Moving Model to GPU")
    if torch.cuda.is_available():
        free_gpu()
        if MODELS_CACHE[id_to_use]["on_gpu"] is False:
            print("Moving model {} to GPU".format(id_to_use))
            MODELS_CACHE[id_to_use]["model"].to("cuda")
            MODELS_CACHE[id_to_use]["on_gpu"] = True

    return MODELS_CACHE[id_to_use]["model"], MODELS_CACHE[id_to_use]["processor"]  # type: ignore
