import torch
from transformers import AutoProcessor

from aaas.model_utils import get_model
from aaas.statics import LANG_MAPPING, MODEL_MAPPING
from aaas.utils import timeit


@timeit
def inference_asr(data_batch, main_lang: str, model_config: str) -> str:
    transcription = []
    model, processor = get_model_and_processor(main_lang, model_config)
    for data in data_batch:
        input_values = processor.feature_extractor(
            data, sampling_rate=16000, return_tensors="pt", truncation=True,
        ).input_features

        if torch.cuda.is_available() and model_config != "large":
            input_values = input_values.to("cuda")
            input_values = input_values.half()
            model = model.to("cuda")
            model = model.half()
            beams = 20
        else:
            beams = 3
        with torch.inference_mode():
            predicted_ids = model.generate(
                input_values,
                max_length=int(((len(data) / 16000) * 12) / 2) + 10,
                use_cache=True,
                no_repeat_ngram_size=3,
                num_beams=beams,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language=LANG_MAPPING[main_lang], task="transcribe"
                ),
            )

        transcription.append(
            processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        )

    return transcription


def get_model_and_processor(lang: str, model_config: str):

    model_id = MODEL_MAPPING[model_config].get(lang, {}).get("name", None)
    if model_id is None:
        lang = "universal"
        model_id = MODEL_MAPPING[model_config][lang]["name"]

    model = MODEL_MAPPING[model_config][lang].get("model", None)
    processor = MODEL_MAPPING[model_config][lang].get("processor", None)
    model_class = MODEL_MAPPING[model_config][lang].get("class", None)

    if processor is None:
        processor = AutoProcessor.from_pretrained(model_id)
        MODEL_MAPPING[model_config][lang]["processor"] = processor

    if model is None:
        model = get_model(model_class, model_id)
        MODEL_MAPPING[model_config][lang]["model"] = model

    return model, processor
