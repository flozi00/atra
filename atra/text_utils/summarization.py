import torch
from atra.model_utils.model_utils import get_model_and_processor
from atra.utils import timeit


def summarize(text, input_lang) -> str:
    text = f"""summarize: {text}"""
    model, tokenizer = get_model_and_processor(input_lang, "summarization")
    inputs = tokenizer(text, return_tensors="pt", max_length=128_000, truncation=True)
    generated_tokens = inference_sum(model, inputs)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return result


@timeit
def inference_sum(model, inputs):
    if torch.cuda.is_available():
        inputs.to("cuda")

    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
        )

    return generated_tokens
