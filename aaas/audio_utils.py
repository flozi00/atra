from transformers import AutoProcessor
from transformers import pipeline

from aaas.model_utils import get_model
from aaas.statics import LANG_MAPPING, MODEL_MAPPING
from aaas.utils import timeit
import torch


@timeit
def inference_asr(data_batch, main_lang: str, model_config: str) -> str:
    transcription = []
    model, processor = get_model_and_processor(main_lang, model_config)
    try:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=LANG_MAPPING[main_lang], task="transcribe"
        )
    except Exception:
        pass

    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
        num_beams=20,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        use_cache=True,
    )
    for data in data_batch:
        transcription.append(
            transcriber(data, chunk_length_s=30, stride_length_s=[15, 0])["text"]
        )

    return transcription


def get_model_and_processor(lang: str, model_config: str):

    model_id = MODEL_MAPPING[model_config].get(lang, {}).get("name", None)
    if model_id is None:
        lang = "universal"
        model_id = MODEL_MAPPING[model_config][lang]["name"]

    model = MODEL_MAPPING[model_config][lang].get("model", None)
    processor = MODEL_MAPPING[model_config][lang].get("processor", None)
    model_class = MODEL_MAPPING[model_config][lang].get("class", None)

    if processor is None:
        processor = AutoProcessor.from_pretrained(model_id)
        MODEL_MAPPING[model_config][lang]["processor"] = processor

    if model is None:
        model = get_model(model_class, model_id)
        MODEL_MAPPING[model_config][lang]["model"] = model

    return model, processor
