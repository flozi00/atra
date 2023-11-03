import torch
from text_to_num.transforms import alpha2digit

from atra.audio_utils.whisper_langs import WHISPER_LANG_MAPPING
from atra.utilities.stats import timeit
import pyloudnorm as pyln
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers import pipeline
import gradio as gr
import warnings
import os

warnings.filterwarnings(action="ignore")

if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0).split(" ")[-1]
else:
    GPU_NAME = "CPU"


pipe = pipeline(
    "automatic-speech-recognition",
    os.getenv("ASR_MODEL", "flozi00/whisper-large-v2-german-cv15"),
    torch_dtype=torch.float16,
    model_kwargs={
        "load_in_4bit": True,
        "use_flash_attention_2": "A" in GPU_NAME or "H" in GPU_NAME or "L" in GPU_NAME,
    },
    batch_size=16,
)
pipe.model.eval()

try:
    pipe.model = torch.compile(pipe.model, backend="onnxrt", mode="reduce-overhead")
except Exception:
    pass


def speech_recognition(data, language, progress=gr.Progress()) -> str:
    if data is None:
        return "", []
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

    progress.__call__(progress=0.8, desc="Transcribing Audio")
    transcription, timestamps = inference_asr(pipe=pipe, data=data, language=language)

    progress.__call__(progress=0.9, desc="Converting to Text")
    try:
        transcription = alpha2digit(
            text=transcription, lang=WHISPER_LANG_MAPPING[language]
        )
        for i in range(len(timestamps)):
            timestamps[i]["text"] = alpha2digit(
                text=timestamps[i]["text"], lang=WHISPER_LANG_MAPPING[language]
            )
    except Exception:
        pass

    return transcription, timestamps


@timeit
def inference_asr(pipe, data, language) -> str:
    generated_ids = pipe(
        data,
        chunk_length_s=30,
        stride_length_s=(10, 0),
        # return_timestamps="word",
        generate_kwargs={
            "task": "transcribe",
            "language": f"<|{WHISPER_LANG_MAPPING[language]}|>",
        },
    )
    return generated_ids["text"], {}
