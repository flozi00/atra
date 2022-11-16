import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import soundfile as sf

model = pretrained.dns64()
wav, sr = torchaudio.load('alex_noisy.mp3')
print(wav, wav.shape, wav[0][1])
wav = convert_audio(wav, sr, model.sample_rate, model.chin)
with torch.no_grad():
    denoised = model(wav[None])[0]

print(denoised)
sf.write("denoised.wav", denoised.data.numpy()[0], samplerate=model.sample_rate, format="WAV", subtype="PCM_24")