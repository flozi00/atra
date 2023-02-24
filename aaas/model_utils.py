from optimum.bettertransformer import BetterTransformer

from aaas.utils import timeit
from aaas.statics import MODEL_MAPPING
from transformers import AutoProcessor
import torch
from peft import PeftModel, PeftConfig
from functools import cache
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from accelerate.utils import get_balanced_memory
from peft.utils import (
    WEIGHTS_NAME,
    PeftType,
    set_peft_model_state_dict,
)
from huggingface_hub import hf_hub_download
import os


class PeftPatch(PeftModel):
    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        r"""
        Args:
        Instantiate a `LoraModel` from a pretrained Lora configuration and weights.
            model (`transformers.PreTrainedModel`):
                The model to be adapted. The model should be initialized with the `from_pretrained` method. from
                `transformers` library.
            model_id (`str`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on
                        huggingface Hub
                    - A path to a directory containing a Lora configuration file saved using the
                        `save_pretrained` method, e.g., ``./my_lora_config_directory/``.
        """
        from peft.mapping import (
            MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
            PEFT_TYPE_TO_CONFIG_MAPPING,
        )

        # load the config
        config = PEFT_TYPE_TO_CONFIG_MAPPING[
            PeftConfig.from_pretrained(model_id).peft_type
        ].from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config)

        # load weights if any
        if os.path.exists(os.path.join(model_id, WEIGHTS_NAME)):
            filename = os.path.join(model_id, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME)
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(filename, map_location="cpu")
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)
        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)
            if model.peft_config.peft_type == PeftType.LORA:
                add_hook_to_module(model.base_model.model, hook)
            else:
                remove_hook_from_submodules(model.prompt_encoder)
                add_hook_to_module(model.base_model, hook)
        return model


@timeit
def get_model(model_class, model_id):
    model = model_class.from_pretrained(
        model_id,
        cache_dir="./model_cache",
        load_in_8bit=False,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    return model


@cache
@timeit
def get_processor(processor_class, model_id):
    processor = processor_class.from_pretrained(model_id)
    return processor


@cache
@timeit
def get_peft_model(peft_model_id, model_class):
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = model_class.from_pretrained(
        peft_config.base_model_name_or_path,
        cache_dir="./model_cache",
    )
    model = PeftPatch.from_pretrained(
        model,
        peft_model_id,
    )
    model = model.eval()
    model.get_base_model().save_pretrained("temp_lora_model", cache_dir="./model_cache")
    del model
    return get_model(model_class, "temp_lora_model")


@cache
@timeit
def get_model_and_processor(lang: str, task: str, config: str):
    # get model id
    model_id = MODEL_MAPPING[task][config].get(lang, {}).get("name", None)
    adapter_id = MODEL_MAPPING[task][config].get(lang, {}).get("adapter_id", None)

    if model_id is None:
        base_model_lang = "universal"
        model_id = MODEL_MAPPING[task][config][base_model_lang]["name"]
    else:
        base_model_lang = lang

    model_class = MODEL_MAPPING[task][config][base_model_lang].get("class", None)

    if adapter_id is not None:
        model = get_peft_model(adapter_id, model_class)
    else:
        model = get_model(model_class, model_id)

    # get processor
    processor_class = MODEL_MAPPING[task][config][base_model_lang].get(
        "processor", AutoProcessor
    )
    processor = get_processor(processor_class, model_id)

    try:
        model = BetterTransformer.transform(model)
    except Exception as e:
        print(e)

    try:
        model = torch.compile(model)
    except Exception as e:
        print(e)

    return model, processor
