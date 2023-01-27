import os
import soundfile as sf
import torch
from transformers.pipelines.audio_utils import ffmpeg_read

demucs_model = "htdemucs"


def seperate_vocal(audio):
    os.makedirs("out", exist_ok=True)
    sf.write("dummy.wav", audio, samplerate=16000)
    os.system(
        f"python -m demucs.separate -n {demucs_model} {'-d cpu' if torch.cuda.is_available() == False else ''} dummy.wav -o out"
    )
    with open(f"./out/{demucs_model}/dummy/vocals.wav", "rb") as f:
        payload = f.read()
    audio = ffmpeg_read(payload, sampling_rate=16000)
    return audio

