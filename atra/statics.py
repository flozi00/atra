from transformers import (
    WhisperForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
)
from diffusers import StableDiffusionPipeline
import os
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
