import torch
from atra.model_utils.model_utils import get_model_and_processor
from atra.utils import timeit
import gradio as gr


def summarize(text, input_lang, progress=gr.Progress()) -> str:
    text = f"""summarize: {text}"""
    model, tokenizer = get_model_and_processor(
        input_lang, "summarization", progress=progress
    )
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    progress.__call__(0.7, "Summarizing Text")
    generated_tokens = inference_sum(model, inputs)
    progress.__call__(0.9, "Converting to Text")
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return result


@timeit
def inference_sum(model, inputs):
    inputs.to(model.device)

    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
        )

    return generated_tokens
