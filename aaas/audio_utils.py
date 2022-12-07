from transformers import AutoProcessor
from aaas.statics import *
from aaas.model_utils import get_model
from aaas.silero_vad import silero_vad
from aaas.utils import timeit
import torch

model_vad, get_speech_timestamps = silero_vad(True)

def preprocess_audio(audio):
    return audio

@timeit
def inference_asr(data_batch, main_lang: str, model_config: str) -> str:
    transcription = []
    model, processor = get_model_and_processor(main_lang, model_config)

    for data in data_batch:
        input_values = processor.feature_extractor(
            data,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=True,
        ).input_features

        with torch.inference_mode():
            predicted_ids = model.generate(
                input_values,
                max_length=int(((len(data) / 16000) * 12) / 2) + 10,
                use_cache=True,
                no_repeat_ngram_size = 5,
                num_beams=1,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language=LANG_MAPPING[main_lang], task="transcribe"
                ),
            )

        transcription.append(
            processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        )

    return transcription

@timeit
def get_model_and_processor(lang: str, model_config: str):
    try:
        model_id = MODEL_MAPPING[model_config][lang]["name"]
    except:
        lang = "universal"
        model_id = MODEL_MAPPING[model_config][lang]["name"]

    model = MODEL_MAPPING[model_config][lang].get("model", None)
    processor = MODEL_MAPPING[model_config][lang].get("processor", None)
    model_class = MODEL_MAPPING[model_config][lang].get("class", None)

    if processor == None:
        processor = AutoProcessor.from_pretrained(model_id)
        MODEL_MAPPING[model_config][lang]["processor"] = processor
        processor.save_pretrained("./model_cache" + model_id.split("/")[-1])

    if model == None:
        model = get_model(model_class, model_id)
        MODEL_MAPPING[model_config][lang]["model"] = model

    return model, processor

@timeit
def batch_audio_by_silence(audio_batch):
    new_batch = []
    tmp_audio = []
    for b in audio_batch:
        if len(tmp_audio) + len(b) < 30 * 16000:
            tmp_audio.extend(b)
        elif len(b) > 28 * 16000:
            new_batch.append(tmp_audio)
            tmp_audio = []
            new_batch.append(b)
        else:
            new_batch.append(tmp_audio)
            tmp_audio = []

    if tmp_audio != []:
        new_batch.append(tmp_audio)

    return new_batch
