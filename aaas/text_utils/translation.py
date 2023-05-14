import torch
from aaas.model_utils import get_model_and_processor
from aaas.statics import LANG_MAPPING


def translate(text, src, dest, model_config: str) -> str:
    src = LANG_MAPPING[src]
    dest = LANG_MAPPING[dest]
    model, tokenizer = get_model_and_processor("universal", "translation", model_config)
    tokenizer.src_lang = src
    input_features = tokenizer(text, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        input_features = input_features.cuda()
    generated_tokens = model.generate(
        inputs=input_features, forced_bos_token_id=tokenizer.get_lang_id(dest)
    )
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return result[0]
