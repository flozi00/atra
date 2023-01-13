import os
import soundfile as sf

demucs_model = "htdemucs"


def seperate_vocal(audio):
    os.makedirs("out", exist_ok=True)
    sf.write("audio.wav", audio, samplerate=16000)
    os.system(f"python -m demucs.separate -n {demucs_model} -d cpu audio.wav -o out")
    return f"./out/{demucs_model}/audio/vocals.wav"

