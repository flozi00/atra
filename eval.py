import datasets
from jiwer import wer, cer
from aaas.audio_utils.asr import inference_asr
import re
from text_to_num import alpha2digit

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
    base_str = alpha2digit(d["sentence"], "de").lower()
    base_str = re.sub(r"[^\w\s]", "", base_str)
    pred_str = inference_asr(d["audio"]["bytes"], "german", "large")
    pred_str = alpha2digit(pred_str, "de")
    pred_str = re.sub(r"[^\w\s]", "", pred_str.lower())
    base.append(base_str)
    predicted.append(pred_str)

    print(wer(base, predicted) * 100, cer(base, predicted) * 100)
    if base_str != pred_str:
        print(base_str, " --> ", pred_str)
