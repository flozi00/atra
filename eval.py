import datasets
from jiwer import wer, cer
from aaas.audio_utils.asr import inference_asr
import re

base = []
predicted = []

ds = (
    datasets.load_dataset(
        "common_voice",
        "de",
        split="test",
        streaming=True,
    )
    ._resolve_features()
    .cast_column("audio", datasets.features.Audio(sampling_rate=16000, decode=False))
)

for d in ds:
    base_str = re.sub(r"[^\w\s]", "", d["sentence"].lower())
    pred_str = inference_asr(d["audio"]["bytes"], "german", "large")
    pred_str = re.sub(r"[^\w\s]", "", pred_str.lower())
    base.append(base_str)
    predicted.append(pred_str)

    print(wer(base, predicted) * 100, cer(base, predicted) * 100)
