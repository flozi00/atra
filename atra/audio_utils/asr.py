import torch
from text_to_num.transforms import alpha2digit

from atra.model_utils.model_utils import get_model_and_processor
from atra.statics import WHISPER_LANG_MAPPING
from atra.utils import timeit
import pyloudnorm as pyln
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.pipelines import AutomaticSpeechRecognitionPipeline
import gradio as gr
import warnings

warnings.filterwarnings(action="ignore")


def speech_recognition(data, language, progress=gr.Progress()) -> str:
    if data is None:
        return ""
    progress.__call__(progress=0.0, desc="Loading Data")
    if isinstance(data, str):
        with open(file=data, mode="rb") as f:
            payload = f.read()

        data = ffmpeg_read(bpayload=payload, sampling_rate=16000)

    progress.__call__(progress=0.1, desc="Normalizing Audio")
    meter = pyln.Meter(rate=16000)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data=data)
    data = pyln.normalize.loudness(
        data=data, input_loudness=loudness, target_loudness=0.0
    )

    progress.__call__(progress=0.2, desc="Loading Model")
    model, processor = get_model_and_processor(
        lang=language, task="asr", progress=progress
    )

    try:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            task="transcribe"
        )
    except Exception as e:
        print("Error in setting forced decoder ids", e)

    progress.__call__(progress=0.7, desc="Initializing Pipeline")
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if torch.cuda.is_available() else -1,
    )

    progress.__call__(progress=0.8, desc="Transcribing Audio")
    transcription = inference_asr(pipe=pipe, data=data)

    progress.__call__(progress=0.9, desc="Converting to Text")
    try:
        transcription = alpha2digit(
            text=transcription, lang=WHISPER_LANG_MAPPING[language]
        )
    except Exception:
        pass

    return transcription


@timeit
def inference_asr(pipe, data) -> str:
    generated_ids = pipe(
        data, chunk_length_s=30, stride_length_s=(10, 0), max_new_tokens=200
    )
    return generated_ids["text"]
