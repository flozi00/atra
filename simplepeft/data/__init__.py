import random
from .AUDIOCollator import ASRDataCollator, TTSDataCollator
from torch.utils.data import DataLoader
from ..data.TEXTCollator import CLMDataCollator, TextTextDataCollator

from ..utils import IS_WINDOWS, Tasks


def get_dataloader(
    task: str, processor: any, datas: any, BATCH_SIZE: int, **kwargs
) -> DataLoader:
    """Get a pytorch dataloader for specified task
    Args: task: task to be used for dataloader
        processor: processor for the task
        datas: data to be used for dataloader
        BATCH_SIZE: batch size to be used for dataloader
        kwargs: additional arguments for the task
    Returns:
        DataLoader for specified task"""
    if task == Tasks.ASR:
        data_collator = ASRDataCollator(
            processor=processor,
            wav_key=kwargs.get(
                "wav_key", "audio"
            ),  # wav_key is a list of keys to get the wav array from the dataset
            locale_key=kwargs.get(
                "locale_key", "locale"
            ),  # locale_key is a key to get the locale (de, en, es, ...) from the dataset
            text_key=kwargs.get(
                "text_key", "sentence"
            ),  # text_key is a key to get the text from the dataset
            max_audio_in_seconds=kwargs.get(
                "max_audio_in_seconds", 10.0
            ),  # max_audio_in_seconds is the maximum length in seconds of the audio
        )
    elif task == Tasks.Text2Text:
        data_collator = TextTextDataCollator(
            tok=processor,
            source_key=kwargs.get(
                "source_key", "source"
            ),  # source_key is a key to get the source text from the dataset
            target_key=kwargs.get(
                "target_key", "target"
            ),  # target_key is a key to get the target text from the dataset
            prefix=kwargs.get(
                "prefix", ""
            ),  # prefix is a prefix to be added to the source text
            max_input_length=kwargs.get(
                "max_input_length", 1024
            ),  # max_input_length is the maximum length in tokens of the input text
            max_output_length=kwargs.get(
                "max_output_length", 1024
            ),  # max_output_length is the maximum length in tokens of the output text
        )
    elif task == Tasks.TEXT_GEN:
        data_collator = CLMDataCollator(
            tok=processor,
            text_key=kwargs.get(
                "text_key", "text"
            ),  # text_key is a key to get the text from the dataset
            max_input_length=kwargs.get(
                "max_input_length", 1024
            ),  # max_input_length is the maximum length in tokens of the input text
        )
    elif task == Tasks.TTS:
        data_collator = TTSDataCollator(
            processor=processor,
            reduction_factor=kwargs.get(
                "reduction_factor", 2
            ),  # reduction_factor is the reduction factor of the model
            wav_key=kwargs.get(
                "wav_key", "audio"
            ),  # wav_key is a list of keys to get the wav array from the dataset
            text_key=kwargs.get(
                "text_key", "sentence"
            ),  # text_key is a key to get the text from the dataset
            speaker_model=kwargs.get(
                "speaker_model", None
            ),  # speaker_model is a model to get the speaker embedding from, needs to be a EncoderClassifier from speechbrain
        )

    datas = datas.shuffle(seed=random.randint(0, 1000))
    dloader = DataLoader(
        datas,
        collate_fn=data_collator,
        batch_size=BATCH_SIZE,
        pin_memory=False,
        num_workers=0,
    )

    return dloader
