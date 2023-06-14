from atra.model_utils.model_utils import get_model_and_processor
from atra.statics import FLORES_LANG_MAPPING
from atra.text_utils.language_detection import classify_language
from atra.utils import timeit
import gradio as gr
from transformers import pipeline
from functools import cache


@cache
def translate(text, src: str = None, dest: str = None, progress=gr.Progress()) -> str:
    if src is None:
        src = classify_language(text)
    if dest is None:
        dest = "English"

    if src == dest:
        return text

    src = FLORES_LANG_MAPPING[src]
    dest = FLORES_LANG_MAPPING[dest]

    model, tokenizer = get_model_and_processor(
        f"{src}-{dest}", "translation", progress=progress
    )
    progress.__call__(0.8, "Translating Text")
    generated_tokens = inference_translate(model, tokenizer, src, dest, text)
    progress.__call__(0.9, "Converting to Text")
    return generated_tokens


@timeit
def inference_translate(model, tokenizer, source, target, text):
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=source,
        tgt_lang=target,
        device=model.device,
        min_new_tokens=int(len(text) / 3.5),
        max_new_tokens=int(len(text) / 2),
    )

    return translator(text)[0]["translation_text"]
