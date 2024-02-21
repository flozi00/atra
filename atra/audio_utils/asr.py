import torch
from text_to_num.transforms import alpha2digit

from atra.audio_utils.whisper_langs import WHISPER_LANG_MAPPING
from atra.utilities.stats import timeit
import warnings
import os
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import torch_tensorrt
from transformers.pipelines.audio_utils import ffmpeg_read

warnings.filterwarnings(action="ignore")

ASR_MODEL = os.getenv("ASR_MODEL", "primeline/whisper-large-v3-german")


processor = WhisperProcessor.from_pretrained(ASR_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    ASR_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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

    transcription = inference_asr(pipe=(model, processor), data=data, language=language)

    try:
        transcription = alpha2digit(
            text=transcription, lang=WHISPER_LANG_MAPPING[language]
        )
    except Exception:
        pass

    return transcription


@timeit
def inference_asr(pipe, data, language) -> str:
    model, processor = pipe
    raw_audio = ffmpeg_read(data, sampling_rate=16_000)
    inputs = processor(
        raw_audio,
        return_tensors="pt",
        truncation=False,
        return_attention_mask=True,
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
            return_timestamps=len(raw_audio) / 16000 >= 30,
            task="transcribe",
            language=f"<|{WHISPER_LANG_MAPPING[language]}|>",
            do_sample=False,
            num_beams=1,
            # assistant_model=assistant_model,
        )

    decoded = processor.batch_decode(result, skip_special_tokens=True)[0]
    return decoded
