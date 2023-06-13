import gradio as gr
import peft
import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor, PreTrainedModel, PreTrainedTokenizer, AutoConfig
from atra.statics import MODEL_MAPPING, PROMPTS
from atra.utils import timeit

MODELS_CACHE = {}
TASK_BLACKLIST = ["embedding", "language-detection"]

def free_gpu(except_model: str, force: bool = False) -> None:
    global MODELS_CACHE
    FREE_GPU_MEM = int(torch.cuda.mem_get_info()[0] / 1024**3)  # in GB
    if FREE_GPU_MEM <= 8 or force is True:
        models_list = list(MODELS_CACHE.keys())
        for model in models_list:
            if MODELS_CACHE[model]["on_gpu"] is True and model != except_model:
                MODELS_CACHE[model]["model"].to("cpu")
                MODELS_CACHE[model]["on_gpu"] = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model(
    model_id: str, model_class: PreTrainedModel, processor_class: PreTrainedTokenizer
) -> bool:
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

        processor = processor_class.from_pretrained(
            pretrained_model_name_or_path=model_id
        )

        conf = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
        )

        if conf.model_type == "wav2vec2":
            conf.update(
                {
                    "vocab_size": len(processor.tokenizer),
                }
            )

        model = model_class.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
            config=conf,
            low_cpu_mem_usage=True,
        )
        try:
            model = peft.PeftModel.from_pretrained(
                model=model,
                model_id=model_id,
            )
            model = model.merge_and_unload()
        except Exception as e:
            print("Error in loading peft model", e)
        MODELS_CACHE[model_id] = {
            "model": model,
            "processor": processor,
            "on_gpu": False,
        }
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
    
    if task == "translation":
        model_id = "Helsinki-NLP/opus-mt-" + lang

    if model_id is None:
        lang = "universal"
        model_id = MODEL_MAPPING[task][lang]["name"]

    model_class = MODEL_MAPPING[task][lang if task != "translation" else "universal"].get("class", None)

    # get processor
    processor_class = MODEL_MAPPING[task][lang if task != "translation" else "universal"].get("processor", AutoProcessor)

    progress.__call__(progress=0.3, desc="Loading Model tensors")
    # load the model
    cached = get_model(
        model_class=model_class, model_id=model_id, processor_class=processor_class
    )

    progress.__call__(progress=0.5, desc="Optimizing Model")
    if cached is False:
        try:
            MODELS_CACHE[model_id]["model"] = BetterTransformer.transform(
                model=MODELS_CACHE[model_id]["model"]
            )
        except Exception as e:
            print("Bettertransformer exception: ", e)
        try:
            if task not in TASK_BLACKLIST:
                MODELS_CACHE[model_id]["model"] = torch.compile(
                    model=MODELS_CACHE[model_id]["model"],
                    mode="max-autotune",
                    backend="onnxrt",
                )
        except Exception as e:
            print("Torch compile exception: ", e)

    progress.__call__(progress=0.6, desc="Moving Model to GPU")
    if torch.cuda.is_available() and task not in TASK_BLACKLIST:
        free_gpu(except_model=model_id)
        FREE_GPU_MEM = int(torch.cuda.mem_get_info()[0] / 1024**3)  # in GB
        if FREE_GPU_MEM >= 8:
            if MODELS_CACHE[model_id]["on_gpu"] is False:
                MODELS_CACHE[model_id]["model"].to("cuda")
                MODELS_CACHE[model_id]["on_gpu"] = True

    return MODELS_CACHE[model_id]["model"], MODELS_CACHE[model_id]["processor"]


def get_prompt(task: str) -> str:
    prompt = PROMPTS[task]

    return prompt
