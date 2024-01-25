import datasets
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
import simplepeft.train.train
from simplepeft.utils import Tasks
import pandas as pd
from unidecode import unidecode
import json
from transformers import Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor

BATCH_SIZE = 2
BASE_MODEL = "w2v-bert-2.0-german-cv16.1"
PEFT_MODEL = "w2v-bert-2.0-german-cv16.1"
TASK = Tasks.ASR
LR = 1e-4
MAX_AUDIO_SEC = 10

simplepeft.train.train.ACCUMULATION_STEPS = 32

vocab_size = None
ctc_processor = None

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


def extract_all_chars(sentences):
  all_text = " ".join(sentences)
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}


def make_ctc_processor(cv_data):
    vocab_train = extract_all_chars(cv_data["sentence"])

    vocab_list = list(set(vocab_train["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    return processor





# generate the dataset from the common voice dataset saved locally and load it as a dataset object
# the dataset is filtered to only contain sentences with more than 5 characters and at least 2 upvotes and no downvotes
# the audio is casted to the Audio feature of the datasets library with a sampling rate of 16000
def get_dataset() -> datasets.Dataset:
    CV_DATA_PATH = "./cv-corpus-16.1-2023-12-06-de/cv-corpus-16.1-2023-12-06/de"

    # filter for durations
    durs = pd.read_table(filepath_or_buffer=f"{CV_DATA_PATH}/clip_durations.tsv")
    durs["clip"] = durs["clip"].astype(dtype=str)
    durs["duration[ms]"] = durs["duration[ms]"].astype(dtype=int)

    mask = (
        (durs["duration[ms]"] <= MAX_AUDIO_SEC * 1000)
        & (durs["duration[ms]"] >= 1000)
    )
    durs = durs.loc[mask]


    # read the data
    df = pd.read_table(filepath_or_buffer=f"{CV_DATA_PATH}/train.tsv")
    df["audio"] = f"{CV_DATA_PATH}/clips/" + df["path"].astype(dtype=str)
    df["down_votes"] = df["down_votes"].astype(dtype=int)
    df["up_votes"] = df["up_votes"].astype(dtype=int)
    df["sentence"] = df["sentence"].astype(dtype=str)

    filter_conditions = {"down_votes": (df["down_votes"] <= 0), "up_votes": (df["up_votes"] >= 2), "sentence": (df["sentence"].str.len() >= 5), "audio_len": (df["path"].isin(durs["clip"]) == True)}

    for key, value in filter_conditions.items():
        before_len = len(df.index)
        df = df.loc[value]
        after_len = len(df.index)
        print(f"Filtered {before_len - after_len} rows for {key}")

    # create the dataset object
    d_sets = datasets.Dataset.from_pandas(df=df)

    d_sets = d_sets.cast_column(
        column="audio", feature=datasets.features.Audio(sampling_rate=16000)
    )

    columns = d_sets.column_names
    for column in columns:
        if column not in ["audio", "sentence"]:
            d_sets = d_sets.remove_columns(column)

    return d_sets


def main():
    cv_data = get_dataset()
    new_column = ["de"] * len(cv_data)
    cv_data = cv_data.add_column("locale", new_column)
    cv_data = cv_data.map(normalize_text)

    #ctc_processor = make_ctc_processor(cv_data)
    #vocab_size = len(ctc_processor.tokenizer)

    model, processor = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
        use_peft=False,
        use_flash_v2=False,
        use_bnb=False,
        lora_depth=64,
        vocab_size=vocab_size,
        processor = ctc_processor
    )

    try:
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
    except Exception:
        pass
    #model.freeze_encoder()

    # get the automatic dataloader for the given task, in this case the default arguments are working for data columns, otherwise they can be specified
    # check the **kwargs in the get_dataloader function in simplepeft/data/main.py for more information
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=cv_data,
        max_audio_in_seconds=MAX_AUDIO_SEC,
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
