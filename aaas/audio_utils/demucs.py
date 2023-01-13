import os
import soundfile as sf


def seperate_vocal(audio):
    os.makedirs("out", exist_ok=True)
    sf.write("audio.wav", audio, samplerate=16000)
    os.system("python -m demucs.separate -n htdemucs_ft audio.wav -o out")
    return "./out/htdemucs_ft/audio/vocals.wav"

