from transformers import (
    WhisperForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
)
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
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
