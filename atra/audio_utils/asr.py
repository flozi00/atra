import torch
from text_to_num.transforms import alpha2digit

from atra.model_utils.model_utils import get_model_and_processor
from atra.statics import LANG_MAPPING
from atra.utils import timeit
import pyloudnorm as pyln
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.pipelines import AutomaticSpeechRecognitionPipeline
import gradio as gr
import warnings

warnings.filterwarnings("ignore")


def speech_recognition(data, language, progress=gr.Progress()) -> str:
    if data is None:
        return ""
    progress.__call__(0.0, "Loading Data")
    if isinstance(data, str):
        with open(data, "rb") as f:
            payload = f.read()

        data = ffmpeg_read(payload, sampling_rate=16000)

    progress.__call__(0.1, "Normalizing Audio")
    meter = pyln.Meter(16000)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)
    data = pyln.normalize.loudness(data, loudness, 0.0)

    progress.__call__(0.2, "Loading Model")
    model, processor = get_model_and_processor(language, "asr", progress=progress)

    try:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            task="transcribe"
        )
    except Exception:
        pass

    progress.__call__(0.7, "Initializing Pipeline")
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,  
        feature_extractor=processor.feature_extractor,  
        device=0 if torch.cuda.is_available() else -1,
    )

    progress.__call__(0.8, "Transcribing Audio")
    transcription = inference_asr(pipe, data)

    progress.__call__(0.9, "Converting to Text")
    try:
        transcription = alpha2digit(transcription, lang=LANG_MAPPING[language])
    except Exception:
        pass

    return transcription


@timeit
def inference_asr(pipe, data) -> str:
    generated_ids = pipe(
        data, chunk_length_s=30, stride_length_s=(10, 0), max_new_tokens=200
    )
    return generated_ids["text"]
