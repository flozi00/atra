import torch
from atra.model_utils.model_utils import get_model_and_processor
from atra.utils import timeit


@timeit
def summarize(text, input_lang) -> str:
    model, tokenizer = get_model_and_processor(input_lang, "summarization")
    input_features = tokenizer(text, return_tensors="pt", max_length=4096).input_ids
    if torch.cuda.is_available():
        input_features = input_features.cuda()
    generated_tokens = model.generate(inputs=input_features)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return result
