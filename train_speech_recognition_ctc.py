import random
import datasets
from simple_nlp.models import get_model
import simple_nlp.train
import json
import os
from transformers import Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor
from torch.utils.data import DataLoader

BATCH_SIZE = 2
BASE_MODEL = os.getenv("BASE_MODEL","flozi00/distilwhisper-german-v2")
OUT_MODEL = os.getenv("OUTPUT_MODEL","distilwhisper-german-canary")

from dataclasses import dataclass
from typing import Dict, List, Union
import torch

from simple_nlp.languages import LANGUAGES
from transformers import AutoProcessor


@dataclass
class ASRDataCollator:
    processor: AutoProcessor
    wav_key: str = os.getenv("AUDIO_PATH","audio")
    locale_key: str = os.getenv("LOCALE_KEY", "de")
    text_key: str = os.getenv("TEXT_KEY", "transkription")
    max_audio_in_seconds: float = float(os.getenv("MAX_AUDIO_IN_SECONDS", 30.0))

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = []
        label_features = []

        for i in range(len(features)):
            try:
                feature = features[i]

                myaudio = feature[self.wav_key]["array"]
                mytext = feature[self.text_key]
            except Exception as e:
                print(e)
                continue

            audio_len = int((len(myaudio) / 16000))
            if audio_len > self.max_audio_in_seconds:
                print("skipping audio")
                continue

            # Extract the text from the feature and normalize it
            mylang = self.locale_key

            # Extract the audio features from the audio
            extracted = self.processor.feature_extractor(
                myaudio,
                sampling_rate=16000,
                return_tensors="pt",
            )

            # check if feature extractor return input_features or input_values
            ft = (
                "input_values"
                if hasattr(extracted, "input_values")
                else "input_features"
            )

            # append to input_features
            input_features.append(
                {
                    ft: getattr(
                        extracted,
                        ft,
                    )[0]
                }
            )

            # set prefix tokens if possible
            try:
                values = list(LANGUAGES.values())
                prefix = mylang if mylang in values else LANGUAGES[mylang]
                self.processor.tokenizer.set_prefix_tokens(prefix)
            except Exception:
                pass

            # append to label_features and tokenize
            label_features.append(
                {"input_ids": self.processor.tokenizer(mytext).input_ids}
            )

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding="longest",
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding="longest",
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def extract_all_chars(sentences):
  all_text = " ".join(sentences)
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}


def make_ctc_processor(cv_data):
    vocab_train = extract_all_chars(cv_data[os.getenv("TEXT_KEY", "transkription")])

    vocab_list = list(set(vocab_train["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(BASE_MODEL)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    return processor


def main():
    cv_data = datasets.load_dataset(os.getenv("DATASET","flozi00/german-canary-asr-0324"), os.getenv("SUBSET","default"), split="train").cast_column(os.getenv("AUDIO_PATH","audio"), datasets.Audio(sampling_rate=16000, decode=True))

    if "w2v" in BASE_MODEL or "wav2vec" in BASE_MODEL:
        ctc_processor = make_ctc_processor(cv_data)
        vocab_size = len(ctc_processor.tokenizer)
    else:
        ctc_processor = None
        vocab_size = None

    model, processor = get_model(
        model_name=BASE_MODEL,
        vocab_size=vocab_size,
        processor = ctc_processor
    )

    try:
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
    except Exception:
        pass


    dataloader = ASRDataCollator(
        processor=processor
    )

    cv_data = cv_data.shuffle(seed=random.randint(0, 1000))
    dloader = DataLoader(
        cv_data,
        collate_fn=dataloader,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=0,
    )

    # start the training
    simple_nlp.train.start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        OUT_MODEL=OUT_MODEL,
    )


if __name__ == "__main__":
    main()