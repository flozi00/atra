import torch
from text_to_num.transforms import alpha2digit

from atra.statics import WHISPER_LANG_MAPPING
from atra.utils import timeit
import pyloudnorm as pyln
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers import pipeline
import gradio as gr
import warnings
from optimum.bettertransformer import BetterTransformer

warnings.filterwarnings(action="ignore")

pipe = None


def speech_recognition(data, language, progress=gr.Progress()) -> str:
    global pipe
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
    if pipe is None:
        pipe = pipeline(
            "automatic-speech-recognition",
            "flozi00/whisper-large-german-lora-cv13",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16,
        )
        pipe.model.eval()
        pipe.model = BetterTransformer.transform(pipe.model)
        pipe.model = torch.compile(pipe.model, backend="onnxrt", mode="max-autotune")

    progress.__call__(progress=0.8, desc="Transcribing Audio")
    transcription = inference_asr(pipe=pipe, data=data, language=language)

    progress.__call__(progress=0.9, desc="Converting to Text")
    try:
        transcription = alpha2digit(
            text=transcription, lang=WHISPER_LANG_MAPPING[language]
        )
    except Exception:
        pass

    return transcription


@timeit
def inference_asr(pipe, data, language) -> str:
    generated_ids = pipe(
        data,
        chunk_length_s=30,
        stride_length_s=(10, 0),
        generate_kwargs={
            "task": "transcribe",
            "language": f"<|{WHISPER_LANG_MAPPING[language]}|>",
        },
    )
    return generated_ids["text"]
