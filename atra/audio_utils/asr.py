import torch
from text_to_num.transforms import alpha2digit

from atra.model_utils.model_utils import get_model_and_processor
from atra.statics import LANG_MAPPING
from atra.utils import timeit
import pyloudnorm as pyln
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.pipelines import AutomaticSpeechRecognitionPipeline

import warnings

warnings.filterwarnings("ignore")


def speech_recognition(data, language) -> str:
    if isinstance(data, str):
        with open(data, "rb") as f:
            payload = f.read()

        data = ffmpeg_read(payload, sampling_rate=16000)

    meter = pyln.Meter(16000)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)
    data = pyln.normalize.loudness(data, loudness, 0.0)

    model, processor = get_model_and_processor(language, "asr")

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids( # type: ignore
        task="transcribe"
    )

    pipe = AutomaticSpeechRecognitionPipeline(model=model, 
            tokenizer=processor.tokenizer, # type: ignore
            feature_extractor=processor.feature_extractor, # type: ignore
            device=0 if torch.cuda.is_available() else -1)

    transcription = inference_asr(pipe, data)

    try:
        transcription = alpha2digit(transcription, lang=LANG_MAPPING[language])
    except Exception:
        pass

    return transcription

@timeit
def inference_asr(pipe, data) -> str:
    generated_ids = pipe(data, chunk_length_s=30,
                     stride_length_s=(10, 0), max_new_tokens=200)
    return generated_ids["text"]