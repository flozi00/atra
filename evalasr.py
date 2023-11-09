import datasets
from jiwer import wer, cer
from atra.audio_utils.asr import speech_recognition
from text_to_num import alpha2digit
from tqdm.auto import tqdm
from unidecode import unidecode
import diff_match_patch as dmp_module
import pandas as pd
import re

base = []
predicted = []

dmp = dmp_module.diff_match_patch()


def normalize_text(batch):
    text = batch["sentence"]
    couples = [
        ("ä", "ae"),
        ("ö", "oe"),
        ("ü", "ue"),
        ("Ä", "Ae"),
        ("Ö", "Oe"),
        ("Ü", "Ue"),
    ]

    # Replace special characters with their ascii equivalent
    for couple in couples:
        text = text.replace(couple[0], f"__{couple[1]}__")
    text = text.replace("ß", "ss")
    text = unidecode(text)

    # Replace the ascii equivalent with the original character after unidecode
    for couple in couples:
        text = text.replace(f"__{couple[1]}__", couple[0])

    text = alpha2digit(text=text, lang="de")

    batch["sentence"] = text
    return batch


def get_dataset() -> datasets.Dataset:
    CV_DATA_PATH = "./cv-corpus-15.0-2023-09-08/de/"
    df = pd.read_table(filepath_or_buffer=f"{CV_DATA_PATH}validated.tsv")
    df["audio"] = f"{CV_DATA_PATH}clips/" + df["path"].astype(dtype=str)
    df["down_votes"] = df["down_votes"].astype(dtype=int)
    df["up_votes"] = df["up_votes"].astype(dtype=int)
    df["sentence"] = df["sentence"].astype(dtype=str)

    mask = (
        (df["down_votes"] <= 0)
        & (df["up_votes"] >= 2)
        & (df["sentence"].str.len() >= 20)
        & (df["sentence"].str.len() <= 100)
    )
    df = df.loc[mask]

    d_sets = datasets.Dataset.from_pandas(df=df)

    d_sets = d_sets.cast_column(
        column="audio", feature=datasets.features.Audio(sampling_rate=16000)
    )

    columns = d_sets.column_names
    for column in columns:
        if column not in ["audio", "sentence"]:
            d_sets = d_sets.remove_columns(column)

    return d_sets


def normalize_str(text):
    text = (
        re.sub(r"[^\w\s]", "", text.strip())
        .lower()
        .replace("-", " ")
        .replace(",", "")
        .replace(".", "")
        .replace("?", "")
        .replace("!", "")
        .replace(":", "")
        .replace(";", "")
        .replace("  ", " ")
        .replace("ph", "f")
        .replace("ß", "ss")
    )

    return text


ds = get_dataset().select(range(100_000))

ds = ds.map(normalize_text, num_proc=8)

pbar = tqdm(ds)

for d in pbar:
    # normalize base transcription
    base_str_orig = d["sentence"]
    base_str_orig = normalize_str(base_str_orig)

    # load audio
    audio_data = d["audio"]["path"]

    # normalize prediction
    pred_str_orig = speech_recognition(audio_data, "german")
    pred_str_orig = normalize_str(pred_str_orig)

    base.append(base_str_orig)
    predicted.append(pred_str_orig)

    if len(base) % 10 == 0:
        pbar.set_description(
            f"WER: {wer(base, predicted) * 100} CER: {cer(base, predicted) * 100}"
        )
