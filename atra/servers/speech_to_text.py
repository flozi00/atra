from pytriton.decorators import batch, group_by_values
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
import base64
import numpy as np
import torch
from text_to_num.transforms import alpha2digit

from atra.utilities.whisper_langs import WHISPER_LANG_MAPPING
from atra.utilities.stats import timeit
import warnings
import os
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import torch_tensorrt # noqa
from transformers.pipelines.audio_utils import ffmpeg_read

warnings.filterwarnings(action="ignore")

ASR_MODEL = os.getenv("ASR_MODEL", "primeline/distil-whisper-large-v3-german")

processor = WhisperProcessor.from_pretrained(ASR_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    ASR_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)

if torch.cuda.is_available():
    model = model.to("cuda:0")

model.eval()
model = torch.compile(
    model, mode="max-autotune", backend="torch_tensorrt", fullgraph=True
)


def speech_recognition(data, language) -> str:
    if data is None:
        return ""
    
    transcriptions = inference_asr(pipe=(model, processor), data=data, language=language)
    for i in range(len(transcriptions)):
        try:
            transcriptions[i] = alpha2digit(
                text=transcriptions[i], lang=WHISPER_LANG_MAPPING[language]
            )
        except Exception:
            pass

    return transcriptions


@timeit
def inference_asr(pipe, data, language) -> str:
    model, processor = pipe
    raw_audio = [ffmpeg_read(audio, sampling_rate=16_000) for audio in data]
    
    inputs = processor(
        raw_audio,
        return_tensors="pt",
        truncation=False,
        padding="longest" if len(raw_audio) > 1 else "max_length", return_attention_mask=True,
        sampling_rate=16_000,
        do_normalize=True,
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda", torch.float16)

    # activate `temperature_fallback` and repetition detection filters and condition on prev text
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            temperature=(
                (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if len(raw_audio) / 16000 > 30 else None
            ),
            return_timestamps=True,
            task="transcribe",
            language=f"<|{WHISPER_LANG_MAPPING[language]}|>",
            do_sample=False,
            num_beams=1,
            # assistant_model=assistant_model,
        )

    decoded = processor.batch_decode(result, skip_special_tokens=True)
    return decoded


@batch
@group_by_values("language")
def _infer_fn(
    language: np.ndarray,
    audio: np.ndarray,
):
    languages = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in language]
    audios = [base64.b64decode(p[0]) for p in audio]

    outputs = []

    transcriptions = speech_recognition(data=audios, language=languages[0])
    for transcription in transcriptions:
        outputs.append(np.char.encode(np.array([transcription]), "utf-8"))

    return {"transcription": np.array(outputs)}


config = TritonConfig(exit_on_error=True, http_port=10_000, grpc_port=10_001, metrics_port=10_002)

triton_server = Triton(config=config)
triton_server.bind(
    model_name="Whisper",
    infer_func=_infer_fn,
    inputs=[
        Tensor(name="language", dtype=np.bytes_, shape=(1,)),
        Tensor(name="audio", dtype=np.bytes_, shape=(1,)),
    ],
    outputs=[
        Tensor(name="transcription", dtype=np.bytes_, shape=(1,)),
    ],
    config=ModelConfig(
        max_batch_size=4,
        batcher=DynamicBatcher(
            max_queue_delay_microseconds=100,
        ),
    ),
    strict=True,
)
