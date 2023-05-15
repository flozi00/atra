import torch
from text_to_num import alpha2digit

from atra.model_utils.model_utils import get_model_and_processor
from atra.statics import LANGUAGE_CODES
from atra.utils import timeit
import pyloudnorm as pyln
from transformers.pipelines.audio_utils import ffmpeg_read

import warnings

warnings.filterwarnings("ignore")


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
def inference_asr(data, model_config: str) -> str:
    if isinstance(data, str):
        with open(data, "rb") as f:
            payload = f.read()

        data = ffmpeg_read(payload, sampling_rate=16000)

    meter = pyln.Meter(16000)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)
    data = pyln.normalize.loudness(data, loudness, 0.0)

    lang = detect_language(data)[0]

    lang = {
        k: v for k, v in sorted(lang.items(), key=lambda item: item[1], reverse=True)
    }.keys()
    lang = list(lang)[0].split("|")[1]

    model, processor = get_model_and_processor(
        LANGUAGE_CODES[lang], "asr", model_config
    )

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        task="transcribe"
    )

    inputs = processor(data, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    if torch.cuda.is_available():
        input_features = input_features.half().cuda()

    with torch.inference_mode():
        generated_ids = model.generate(
            inputs=input_features, max_new_tokens=448, num_beams=5
        )

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()

    try:
        transcription = alpha2digit(transcription, lang=lang)
    except Exception:
        pass

    return transcription, LANGUAGE_CODES[lang]
