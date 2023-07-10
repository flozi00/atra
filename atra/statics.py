from transformers import (
    WhisperForConditionalGeneration,
    AutoProcessor,
)
from atra.audio_utils.whisper_langs import WHISPER_LANG_MAPPING, WHISPER_LANGUAGE_CODES

WHISPER_LANGUAGE_CODES
WHISPER_LANG_MAPPING

MODEL_MAPPING = {
    "asr": {
        "german": {
            "name": "flozi00/whisper-large-german-lora-cv13",
            "class": WhisperForConditionalGeneration,
            "processor": AutoProcessor,
        },
        "universal": {
            "name": "openai/whisper-large-v2",
            "class": WhisperForConditionalGeneration,
            "processor": AutoProcessor,
        },
    },
}
