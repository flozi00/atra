import GPUtil
import gradio as gr
import peft
import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor, PreTrainedModel, PreTrainedTokenizer, AutoConfig

from atra.model_utils.unlimiformer import Unlimiformer
from atra.statics import MODEL_MAPPING
from atra.utils import timeit

MODELS_CACHE = {}


def free_gpu(except_model: str) -> None:
    global MODELS_CACHE
    gpus = GPUtil.getGPUs()
    for gpu_num in range(len(gpus)):
        gpu: GPUtil.GPU = gpus[gpu_num]

        if gpu.memoryUtil * 100 > 60:
            models_list = list(MODELS_CACHE.keys())
            for model in models_list:
                if MODELS_CACHE[model]["on_gpu"] is True and model != except_model:
                    MODELS_CACHE[model]["model"].to("cpu")
                    MODELS_CACHE[model]["on_gpu"] = False
                    print("Model {} moved to CPU".format(model))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model(model_id: str, model_class: PreTrainedModel, processor_class: PreTrainedTokenizer) -> bool:
    global MODELS_CACHE
    cached = MODELS_CACHE.get(model_id, None)
    base_model_name = model_id
    if cached is None:
        try:
            peft_config = peft.PeftConfig.from_pretrained(
                pretrained_model_name_or_path=model_id
            )
            base_model_name = peft_config.base_model_name_or_path
        except Exception as e:
            print("Error in loading peft config", e)

        processor = processor_class.from_pretrained(pretrained_model_name_or_path=model_id)
        
        conf = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
            cache_dir="./model_cache",
        )

        if conf.model_type == "wav2vec2":
            conf.update({"vocab_size": len(processor.tokenizer),})
        
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
            cache_dir="./model_cache",
            config=conf,
        )
        try:
            model = peft.PeftModel.from_pretrained(
                model=model,
                model_id=model_id,
                cache_dir="./model_cache",
            )
            model = model.merge_and_unload()  
        except Exception as e:
            print("Error in loading peft model", e)
        MODELS_CACHE[model_id] = {"model": model, "processor":processor ,"on_gpu": False}
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
    progress.__call__(progress=0.25, desc="Loading Model Information")
    model_id = MODEL_MAPPING[task].get(lang, {}).get("name", None)
    if model_id is None:
        lang = "universal"
        model_id = MODEL_MAPPING[task][lang]["name"]

    model_class = MODEL_MAPPING[task][lang].get("class", None)

    # get processor
    processor_class = MODEL_MAPPING[task][lang].get(
        "processor", AutoProcessor
    )

    progress.__call__(progress=0.3, desc="Loading Model tensors")
    # load the model
    cached = get_model(model_class=model_class, model_id=model_id, processor_class=processor_class)

    progress.__call__(progress=0.5, desc="Optimizing Model")
    if cached is False:
        if MODELS_CACHE[model_id]["model"].config.model_type == "t5":
            print("Converting T5 model to Unlimiformer")
            MODELS_CACHE[model_id]["model"] = Unlimiformer.convert_model(
                model=MODELS_CACHE[model_id]["model"]
            )
            MODELS_CACHE[model_id]["model"].eval()
        else:
            try:
                MODELS_CACHE[model_id]["model"] = BetterTransformer.transform(model=MODELS_CACHE[model_id]["model"])  
            except Exception as e:
                print("Bettertransformer exception: ", e)
            try:
                MODELS_CACHE[model_id]["model"] = torch.compile(
                    model=MODELS_CACHE[model_id]["model"],
                    mode="max-autotune",
                    backend="onnxrt",
                )
            except Exception as e:
                print("Torch compile exception: ", e)

    progress.__call__(progress=0.6, desc="Moving Model to GPU")
    if torch.cuda.is_available():
        free_gpu(except_model=model_id)
        if MODELS_CACHE[model_id]["on_gpu"] is False:
            print("Moving model {} to GPU".format(model_id))
            MODELS_CACHE[model_id]["model"].to("cuda")
            MODELS_CACHE[model_id]["on_gpu"] = True

    return MODELS_CACHE[model_id]["model"], MODELS_CACHE[model_id]["processor"]  
