from transformers import (
    WhisperForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
)
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
            "name": "flozi00/whisper-large-v2-german-cv13",
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
    "chat": {
        "universal": {
            "name": "flozi00/OpenAssistant-SFT-7-Llama-30B-4-bits-autogptq",
            "class": AutoModelForCausalLM,
            "processor": AutoTokenizer,
        },
    },
    "embedding": {
        "universal": {
            "name": "intfloat/e5-base-v2",
            "class": AutoModel,
            "processor": AutoTokenizer,
        },
    },
}


TASK_MAPPING = {
    "asr": ["start", "end", "language"],
    "translation": ["source", "target"],
    "summarization": ["long_text", "short_text"],
    "question-answering": ["question", "lang"],
}

SEARCH_BACKENDS = os.getenv("SEARCH_BACKENDS", "gruble.de,searx.fmac.xyz,search.sapti.me").split(",")

HUMAN_PREFIX = "<|prompter|>"
ASSISTANT_PREFIX = "<|assistant|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"

PROMPTS = {
    "question-answering": """Context: {text}
Task: Answer the following task based on the facts given in the context. Give reasons for your answer.
Question: {question}""",
}