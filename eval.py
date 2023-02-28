import datasets
from jiwer import wer, cer
from aaas.audio_utils.asr import inference_asr
import re
from text_to_num import alpha2digit
from transformers.pipelines.audio_utils import ffmpeg_read


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
    # normalize base transcription
    base_str_orig = alpha2digit(d["sentence"], "de")
    base_str = re.sub(r"[^\w\s]", "", base_str_orig).lower()

    # load audio
    audio_data = ffmpeg_read(d["audio"]["bytes"], sampling_rate=16000)

    # normalize prediction
    pred_str = inference_asr(audio_data, "german", "large")
    pred_str_orig = alpha2digit(pred_str, "de")
    pred_str = re.sub(r"[^\w\s]", "", pred_str_orig).lower()

    # append to lists
    base.append(base_str)
    predicted.append(pred_str)

    # print results
    print(wer(base, predicted) * 100, cer(base, predicted) * 100)
    if base_str != pred_str:
        print(f"Base: {base_str_orig} ({base_str})")
        print(f"Pred: {pred_str_orig} ({pred_str})")
        print()

    if len(base) > 100:
        break
