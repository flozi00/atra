import datasets
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
import simplepeft.train.train
from simplepeft.utils import Tasks
import pandas as pd
from unidecode import unidecode

BATCH_SIZE = 16
BASE_MODEL = "whisper-large-v3-german-cv15-v1"
PEFT_MODEL = "whisper-large-v3-german-cv15-v2"
TASK = Tasks.ASR
LR = 1e-5

simplepeft.train.train.ACCUMULATION_STEPS = 4


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

    batch["sentence"] = text
    return batch


# generate the dataset from the common voice dataset saved locally and load it as a dataset object
# the dataset is filtered to only contain sentences with more than 5 characters and at least 2 upvotes and no downvotes
# the audio is casted to the Audio feature of the datasets library with a sampling rate of 16000
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
        & (df["sentence"].str.len() >= 5)
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


def get_hf_datasets() -> datasets.Dataset:
    ds = datasets.load_dataset("facebook/voxpopuli", "de", split="train")
    ds = ds.rename_column("raw_text", "sentence")

    columns = ds.column_names
    for column in columns:
        if column not in ["audio", "sentence"]:
            ds = ds.remove_columns(column)

    return ds


def main():
    cv_data = get_dataset()
    # vox_data = get_hf_datasets()
    # cv_data = datasets.concatenate_datasets([cv_data, vox_data])

    new_column = ["de"] * len(cv_data)
    cv_data = cv_data.add_column("locale", new_column)

    cv_data = cv_data.map(normalize_text, num_proc=8)
    model, processor = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
        use_peft=True,
        use_flash_v2=True,
        use_bnb=True,
        lora_depth=512,
    )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # get the automatic dataloader for the given task, in this case the default arguments are working for data columns, otherwise they can be specified
    # check the **kwargs in the get_dataloader function in simplepeft/data/main.py for more information
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=cv_data,
        max_audio_in_seconds=28,
        BATCH_SIZE=BATCH_SIZE,
    )

    # start the training
    simplepeft.train.train.start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
    )


if __name__ == "__main__":
    main()
