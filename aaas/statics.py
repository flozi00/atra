from transformers import AutoModelForSpeechSeq2Seq
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

MODEL_MAPPING = {
    "german": {"name": "flozi00/whisper-base-german", "class": ORTModelForSpeechSeq2Seq},
    "universal": {"name": "openai/whisper-large", "class": ORTModelForSpeechSeq2Seq},
}
LANG_MAPPING = {"german": "de", "english": "en", "french": "fr", "spanish": "es"}
