from transformers import AutoModelForSpeechSeq2Seq
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

MODEL_MAPPING = {
    "small":{
        "german": {"name": "flozi00/whisper-small-german", "class": ORTModelForSpeechSeq2Seq},
        "english": {"name": "flozi00/whisper-small.en.onnx", "class": ORTModelForSpeechSeq2Seq},
        "universal": {"name": "flozi00/whisper-small.onnx", "class": ORTModelForSpeechSeq2Seq},
    },
    "medium":{
        "german": {"name": "flozi00/whisper-medium-german", "class": ORTModelForSpeechSeq2Seq},
        "english": {"name": "openai/whisper-medium.en", "class": ORTModelForSpeechSeq2Seq},
        "universal": {"name": "openai/whisper-medium", "class": ORTModelForSpeechSeq2Seq},
    },
    "large":{
        "german": {"name": "flozi00/whisper-large-german", "class": ORTModelForSpeechSeq2Seq},
        "universal": {"name": "openai/whisper-large", "class": ORTModelForSpeechSeq2Seq},
    },
}
LANG_MAPPING = {"german": "de", "english": "en", "french": "fr", "spanish": "es"}
