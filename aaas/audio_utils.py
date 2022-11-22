from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torch.quantization
import torch.nn as nn
from optimum.bettertransformer import BetterTransformer
from aaas.backend_utils import inference_only
from aaas.statics import *
from aaas.backend_utils import ipex_optimizer
import intel_extension_for_pytorch as ipex

if inference_only == False:
    from aaas.silero_vad import silero_vad

    model_vad, get_speech_timestamps = silero_vad(True)


def inference_denoise(audio):
    return audio


def inference_asr(data_batch, main_lang: str, model_config: str) -> str:
    transcription = []
    if model_config == "multilingual":
        model, processor = get_model_and_processor("universal")
    else:
        model, processor = get_model_and_processor(main_lang)

    for data in data_batch:
        input_values = processor.feature_extractor(
            data,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=True,
        ).input_features

        with torch.inference_mode():
            if torch.cuda.is_available():
                input_values = input_values.to("cuda").half()
            predicted_ids = model.generate(
                input_values,
                max_length=int(((len(data) / 16000) * 12) / 2) + 10,
                use_cache=True,
                no_repeat_ngram_size=1,
                num_beams=2,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language=LANG_MAPPING[main_lang], task="transcribe"
                ),
            )

        transcription.append(
            processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        )

    return transcription


def get_model_and_processor(lang: str):
    try:
        model_id = MODEL_MAPPING[lang]["name"]
    except:
        lang = "universal"
        model_id = MODEL_MAPPING[lang]["name"]

    model = MODEL_MAPPING[lang].get("model", None)
    processor = MODEL_MAPPING[lang].get("processor", None)

    if processor == None:
        processor = WhisperProcessor.from_pretrained(model_id)
        MODEL_MAPPING[lang]["processor"] = processor

    if model == None:
        model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
        model = BetterTransformer.transform(model)

        if torch.cuda.is_available():
            model = model.to("cuda").half()
        elif ipex_optimizer == True:
            model = ipex.optimize(model)
        else:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )

        MODEL_MAPPING[lang]["model"] = model

    return model, processor


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
