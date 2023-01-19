import os
import soundfile as sf
import torch

demucs_model = "htdemucs"


def seperate_vocal(audio):
    os.makedirs("out", exist_ok=True)
    sf.write("dummy.wav", audio, samplerate=16000)
    os.system(
        f"python -m demucs.separate -n {demucs_model} {'-d cpu' if torch.cuda.is_available() == False else ''} dummy.wav -o out"
    )
    return f"./out/{demucs_model}/dummy/vocals.wav"

