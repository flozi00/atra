import datasets
from jiwer import wer, cer
from aaas.audio_utils.asr import inference_asr
import re
from text_to_num import alpha2digit
from tqdm.auto import tqdm
import language_tool_python
import language_tool_python.download_lt

language_tool_python.download_lt.LATEST_VERSION = "stable"
tool = language_tool_python.LanguageTool("de-DE")

base = []
predicted = []

ds = datasets.load_dataset(
    "common_voice",
    "de",
    split="test",
).cast_column("audio", datasets.features.Audio(sampling_rate=16000, decode=False))

ds = ds.filter(lambda x: x["down_votes"] <= 0)
ds = ds.filter(lambda x: x["up_votes"] <= 2)

ds = ds.cast_column("audio", datasets.features.Audio(sampling_rate=16000, decode=True))

for d in tqdm(ds):
    # normalize base transcription
    base_str_orig = alpha2digit(d["sentence"], "de")
    base_str_orig = tool.correct(base_str_orig)
    base_str = (
        re.sub(r"[^\w\s]", "", base_str_orig).lower().replace("-", " ").replace(",", "")
    )

    try:
        # load audio
        audio_data = d["audio"]["array"]

        # normalize prediction
        pred_str = inference_asr(audio_data, "german", "large")
        pred_str = tool.correct(pred_str)
        pred_str_orig = alpha2digit(pred_str, "de")
        pred_str = (
            re.sub(r"[^\w\s]", "", pred_str_orig)
            .lower()
            .replace("-", " ")
            .replace(",", "")
        )

        # append to lists
        base.append(
            pred_str
            if base_str.replace(" ", "") == pred_str.replace(" ", "")
            else base_str
        )
        predicted.append(pred_str)

        # print results
        if base_str != pred_str:
            print(f"Base: {base_str}")
            print(f"Pred: {pred_str}")
            print(wer(base, predicted) * 100, cer(base, predicted) * 100)
            print()
    except Exception as e:
        print(e)

print(wer(base, predicted) * 100, cer(base, predicted) * 100)
