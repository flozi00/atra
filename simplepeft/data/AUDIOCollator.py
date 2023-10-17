from dataclasses import dataclass, field
from typing import Dict, List, Union
import torch

from ..languages import LANGUAGES
from transformers import AutoProcessor


@dataclass
class ASRDataCollator:
    processor: AutoProcessor
    wav_key: list = field(default_factory=list)
    locale_key: str = "locale"
    text_key: str = "sentence"
    max_audio_in_seconds: float = 10.0

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = []
        label_features = []

        for i in range(len(features)):
            feature = features[i]

            myaudio = feature[self.wav_key]["array"]
            mytext = feature[self.text_key]

            audio_len = int((len(myaudio) / 16000))
            if audio_len > self.max_audio_in_seconds:
                print("skipping audio")
                continue

            # Extract the text from the feature and normalize it
            mylang = feature[self.locale_key]

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


@dataclass
class TTSDataCollator:
    processor: AutoProcessor
    reduction_factor: int
    speaker_model: any
    wav_key: list = field(default_factory=list)
    text_key: str = "sentence"

    def create_speaker_embedding(self, waveform):
        # Create a torch tensor from the waveform
        with torch.no_grad():
            speaker_embeddings = self.speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(
                speaker_embeddings, dim=2
            )
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids, label_features, speaker_features = [], [], []

        # Extract the audio from the feature even its nested
        for feature in features:
            myaudio = feature[self.wav_key]["array"]

            mytext = feature[self.text_key]
            # feature extraction and tokenization
            example = self.processor(
                text=mytext,
                audio_target=myaudio,
                sampling_rate=16000,
                return_attention_mask=False,
            )

            # strip off the batch dimension
            example["labels"] = example["labels"][0]

            # use SpeechBrain to obtain x-vector
            example["speaker_embeddings"] = self.create_speaker_embedding(myaudio)

            input_ids.append({"input_ids": example["input_ids"]})
            label_features.append({"input_values": example["labels"]})
            speaker_features.append(example["speaker_embeddings"])

        # collate the inputs and targets into a batch
        batch = self.processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if self.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [length - length % self.reduction_factor for length in target_lengths]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch
