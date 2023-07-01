from transformers import (
    WhisperForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
)
from diffusers import StableDiffusionPipeline
import os
from atra.audio_utils.whisper_langs import WHISPER_LANG_MAPPING, WHISPER_LANGUAGE_CODES
from atra.text_utils.flores_langs import FLORES_LANG_MAPPING, FLORES_LANGUAGE_CODES

WHISPER_LANGUAGE_CODES
WHISPER_LANG_MAPPING

FLORES_LANGUAGE_CODES
FLORES_LANG_MAPPING

MODEL_MAPPING = {
    "language-detection": {
        "universal": {
            "name": "juliensimon/xlm-v-base-language-id",
            "class": AutoModelForSequenceClassification,
            "processor": AutoTokenizer,
        },
    },
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
    "translation": {
        "universal": {
            "name": "facebook/nllb-200-distilled-1.3B",
            "class": AutoModelForSeq2SeqLM,
            "processor": AutoTokenizer,
        }
    },
    "embedding": {
        "universal": {
            "name": "intfloat/multilingual-e5-base",
            "class": AutoModel,
            "processor": AutoTokenizer,
        },
    },
    "diffusion": {
        "photo-real": {
            "name": "dreamlike-art/dreamlike-photoreal-2.0",
            "class": StableDiffusionPipeline,
            "processor": None,
        },
        "openjourney": {
            "name": "prompthero/openjourney",
            "class": StableDiffusionPipeline,
            "processor": None,
        },
        "sd2.1": {
            "name": "stabilityai/stable-diffusion-2-1",
            "class": StableDiffusionPipeline,
            "processor": None,
        },
    },
}

SEARCH_BACKENDS = os.getenv(
    "SEARCH_BACKENDS", "gruble.de,searx.fmac.xyz,search.sapti.me"
).split(",")

HUMAN_PREFIX = "<|prompter|>"
ASSISTANT_PREFIX = "<|assistant|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"

PROMPTS = {
    "question-answering": HUMAN_PREFIX
    + """Context: {text}
Task: Answer the following task based on the facts given in the context.
Question: {question}"""
    + END_OF_TEXT_TOKEN,
}
