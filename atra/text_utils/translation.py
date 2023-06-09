from atra.model_utils.model_utils import get_model_and_processor
from atra.statics import FLORES_LANG_MAPPING
from atra.utils import timeit
import gradio as gr
from transformers import pipeline

def translate(text, src, dest, progress=gr.Progress()) -> str:
    if src == dest:
        return text

    src = FLORES_LANG_MAPPING[src]
    dest = FLORES_LANG_MAPPING[dest]

    model, tokenizer = get_model_and_processor(
        "universal", "translation", progress=progress
    )
    progress.__call__(0.8, "Translating Text")
    generated_tokens = inference_translate(
        model, tokenizer, src, dest, text
    )
    progress.__call__(0.9, "Converting to Text")
    return generated_tokens


@timeit
def inference_translate(model, tokenizer, source, target, text):
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source, tgt_lang=target, device=model.device)

    return translator(text)[0]['translation_text']