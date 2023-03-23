from functools import cache

import torch
from text_to_num import alpha2digit
from transformers.pipelines import AutomaticSpeechRecognitionPipeline as pipeline

from aaas.model_utils import get_model_and_processor
from aaas.statics import LANG_MAPPING
from aaas.utils import timeit


@cache
def get_pipeline(main_lang: str, model_config: str):
    model, processor = get_model_and_processor(main_lang, "asr", model_config)
    transcriber = pipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        ignore_warning=True,
        chunk_length_s=30,
        stride_length_s=[15, 0],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
        return_timestamps=False,
    )
    return transcriber


@timeit
def inference_asr(data, main_lang: str, model_config: str) -> str:
    transcriber = get_pipeline(main_lang, model_config)

    transcription = transcriber(
        data,
        generate_kwargs={
            "task": "transcribe",
            "language": f"<|{LANG_MAPPING[main_lang]}|>",
            "use_cache": True,
            "num_beams": 1,
            "max_new_tokens": int((len(data) / 16000) * 10) + 10,
        },
    )["text"].strip()

    transcription = alpha2digit(transcription, lang=LANG_MAPPING[main_lang])

    return transcription
