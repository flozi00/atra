import torch
from text_to_num import alpha2digit
from transformers.pipelines import AutomaticSpeechRecognitionPipeline as pipeline
from aaas.audio_utils.demucs import seperate_vocal

from aaas.model_utils import get_model_and_processor
from aaas.statics import LANGUAGE_CODES
from aaas.utils import timeit

lang_model, lang_processor = get_model_and_processor(
    "universal", "asr", "small", activate_cache=False
)


def detect_language(data) -> list:
    possible_languages = list(LANGUAGE_CODES.keys())

    input_features = lang_processor(
        data, sampling_rate=16000, return_tensors="pt"
    ).input_features

    if torch.cuda.is_available():
        input_features = input_features.half()

    # hacky, but all language tokens and only language tokens are 6 characters long
    language_tokens = [
        t for t in lang_processor.tokenizer.additional_special_tokens if len(t) == 6
    ]
    language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]

    language_token_ids = lang_processor.tokenizer.convert_tokens_to_ids(language_tokens)

    # 50258 is the token for transcribing
    logits = lang_model(
        input_features,
        decoder_input_ids=torch.tensor(
            [[50258] for _ in range(input_features.shape[0])]
        ),
    ).logits
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float("inf")

    logits = logits.cpu().to(torch.float32)
    output_probs = logits.softmax(dim=-1)
    return [
        {
            lang: output_probs[input_idx, 0, token_id].item()
            for token_id, lang in zip(language_token_ids, language_tokens)
        }
        for input_idx in range(logits.shape[0])
    ]


@timeit
def get_pipeline(main_lang: str, model_config: str):
    model, processor = get_model_and_processor(main_lang, "asr", model_config)
    transcriber = pipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        ignore_warning=True,
        chunk_length_s=30,
        stride_length_s=[15, 0],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
        return_timestamps=False,
    )
    return transcriber, processor


@timeit
def inference_asr(data, model_config: str, is_reclamation: bool) -> str:
    if is_reclamation is True:
        data = seperate_vocal(data)

    lang = detect_language(data)[0]

    lang = {
        k: v for k, v in sorted(lang.items(), key=lambda item: item[1], reverse=True)
    }.keys()
    lang = list(lang)[0].split("|")[1]

    transcriber, processor = get_pipeline(LANGUAGE_CODES[lang], model_config)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=lang, task="transcribe"
    )

    transcription = transcriber(
        data,
        generate_kwargs={
            "forced_decoder_ids": forced_decoder_ids,
            "use_cache": True,
            "num_beams": 1,
            "max_new_tokens": int((len(data) / 16000) * 10) + 10,
        },
        batch_size=1,
    )["text"].strip()

    try:
        transcription = alpha2digit(transcription, lang=lang)
    except Exception:
        pass

    if torch.cuda.is_available():
        transcriber.model.cpu()
        torch.cuda.empty_cache()

    return transcription, LANGUAGE_CODES[lang]
