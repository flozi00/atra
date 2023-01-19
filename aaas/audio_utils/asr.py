from transformers import pipeline

from aaas.model_utils import get_model_and_processor
from aaas.statics import LANG_MAPPING
from aaas.utils import timeit
import torch
import librosa


@timeit
def inference_asr(data, main_lang: str, model_config: str) -> list:
    model, processor = get_model_and_processor(main_lang, "asr", model_config)
    try:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=LANG_MAPPING[main_lang], task="transcribe"
        )
    except Exception:
        pass

    if torch.cuda.is_available():
        model = model.half()

    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ignore_warning=True,
    )

    audio = librosa.effects.time_stretch(data, 0.75)
    return transcriber(audio, chunk_length_s=30, stride_length_s=[15, 0])["text"]
