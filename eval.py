import datasets
from jiwer import wer, cer
from aaas.audio_utils.asr import inference_asr
import re
from text_to_num import alpha2digit
from tqdm.auto import tqdm

base = []
predicted = []

ds = datasets.load_dataset(
    "common_voice",
    "de",
    split="test",
).cast_column("audio", datasets.features.Audio(sampling_rate=16000, decode=False))

ds = (
    ds.filter(lambda x: x["down_votes"] <= 0)
    .filter(lambda x: x["up_votes"] <= 2)
    .cast_column("audio", datasets.features.Audio(sampling_rate=16000, decode=True))
)

for d in tqdm(ds):
    # normalize base transcription
    base_str_orig = alpha2digit(d["sentence"], "de")
    base_str = (
        re.sub(r"[^\w\s]", "", base_str_orig.strip())
        .lower()
        .replace("-", " ")
        .replace(",", "")
        .replace("  ", " ")
        .replace("ph", "f")
        .replace("ß", "ss")
    )

    try:
        # load audio
        audio_data = d["audio"]["array"]

        # normalize prediction
        pred_str_orig = alpha2digit(inference_asr(audio_data, "german", "large"), "de")
        pred_str = (
            re.sub(r"[^\w\s]", "", pred_str_orig.strip())
            .lower()
            .replace("-", " ")
            .replace(",", "")
            .replace("  ", " ")
            .replace("ph", "f")
            .replace("ß", "ss")
        )

        # append to lists
        base.append(
            pred_str
            if base_str.replace(" ", "") == pred_str.replace(" ", "")
            else base_str
        )
        predicted.append(pred_str)

        # print results
        if base_str.replace(" ", "") != pred_str.replace(" ", ""):
            print(f"Base: {base_str}")
            print(f"Pred: {pred_str}")
            print(wer(base, predicted) * 100, cer(base, predicted) * 100)
            print()
    except Exception as e:
        print(e)

    if len(base) > 100:
        break

print(wer(base, predicted) * 100, cer(base, predicted) * 100)
