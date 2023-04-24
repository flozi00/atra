import os

import soundfile as sf
from transformers.pipelines.audio_utils import ffmpeg_read

from aaas.utils import timeit

demucs_model = "htdemucs_ft"


@timeit
def seperate_vocal(audio):
    os.makedirs("out", exist_ok=True)
    sf.write("dummy.wav", audio, samplerate=16000)
    os.system(f"python -m demucs.separate -n {demucs_model} -d cpu dummy.wav -o out")
    with open(f"./out/{demucs_model}/dummy/vocals.wav", "rb") as f:
        payload = f.read()
    audio = ffmpeg_read(payload, sampling_rate=16000)
    return audio
