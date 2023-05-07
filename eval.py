import re

import datasets
import diff_match_patch as dmp_module
from jiwer import cer, wer
from text_to_num import alpha2digit
from tqdm.auto import tqdm
from unidecode import unidecode

from aaas.audio_utils.asr import inference_asr

base = []
predicted = []

dmp = dmp_module.diff_match_patch()


def get_cv():
    ds = datasets.load_dataset(
        "common_voice",
        "de",
        split="test",
    ).cast_column("audio", datasets.features.Audio(sampling_rate=16000, decode=False))

    ds = (
        ds.filter(lambda x: x["down_votes"] == 0)
        .filter(lambda x: x["up_votes"] >= 2)
        .filter(lambda x: len(x["sentence"]) >= 16)
        .sort("up_votes", reverse=False)
        .cast_column("audio", datasets.features.Audio(sampling_rate=16000, decode=True))
    )

    return ds


def get_fleurs():
    ds = datasets.load_dataset("google/fleurs", "de_de", split="test").cast_column(
        "audio", datasets.features.Audio(sampling_rate=16000, decode=True)
    )
    ds = ds.rename_column("transcription", "sentence")

    return ds


ds = get_fleurs()

for d in tqdm(ds):
    # normalize base transcription
    base_str_orig = unidecode(alpha2digit(d["sentence"], "de"))
    base_str = (
        re.sub(r"[^\w\s]", "", base_str_orig.strip())
        .lower()
        .replace("-", " ")
        .replace(",", "")
        .replace("  ", " ")
        .replace("ph", "f")
        .replace("ß", "ss")
    )

    # load audio
    audio_data = d["audio"]["array"]

    # normalize prediction
    pred_str_orig = unidecode(
        alpha2digit(inference_asr(audio_data, "small", False)[0], "de")
    )
    pred_str = (
        re.sub(r"[^\w\s]", "", pred_str_orig.strip())
        .lower()
        .replace("-", " ")
        .replace(",", "")
        .replace("  ", " ")
        .replace("ph", "f")
        .replace("ß", "ss")
    )

    diff_score = 0
    diff = dmp.diff_main(base_str.replace(" ", ""), pred_str.replace(" ", ""))
    dmp.diff_cleanupSemantic(diff)
    for d in diff:
        if d[0] != 0:
            if len(d[1]) > 1:
                diff_score += 1

    # append to lists
    base.append(pred_str if diff_score == 0 else base_str)
    predicted.append(pred_str)

    # print results
    if diff_score != 0:
        diff = dmp.diff_main(base_str, pred_str)
        dmp.diff_cleanupSemantic(diff)
        print()
        print(diff)
        print("normalized", wer(base, predicted) * 100, cer(base, predicted) * 100)
        print()

    if len(base) > 100000:
        break

print(wer(base, predicted) * 100, cer(base, predicted) * 100)
