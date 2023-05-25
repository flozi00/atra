import torch
from atra.model_utils.model_utils import get_model_and_processor
from atra.statics import LANG_MAPPING
from atra.utils import timeit
import gradio as gr


def translate(text, src, dest, progress=gr.Progress()) -> str:
    if src == dest:
        return text
    src = LANG_MAPPING[src]
    dest = LANG_MAPPING[dest]
    model, tokenizer = get_model_and_processor(
        "universal", "translation", progress=progress
    )
    tokenizer.src_lang = src
    progress.__call__(0.7, "Tokenizing Text")
    input_features = tokenizer(text, return_tensors="pt")
    progress.__call__(0.8, "Translating Text")
    generated_tokens = inference_translate(
        model, input_features, tokenizer.get_lang_id(dest)
    )
    progress.__call__(0.9, "Converting to Text")
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return result[0]


@timeit
def inference_translate(model, input_features, forced_bos_id):
    if torch.cuda.is_available():
        input_features.to("cuda")
    with torch.inference_mode():
        generated_tokens = model.generate(
            **input_features, forced_bos_token_id=forced_bos_id
        )

    return generated_tokens
