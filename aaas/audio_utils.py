import subprocess
import numpy as np
from aaas.silero_vad import silero_vad
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_MAPPING = {
    "german": {"name": "aware-ai/whisper-base-european"},
    "universal": {"name": "openai/whisper-large"},
}
LANG_MAPPING = {"german": "de"}


def get_model_and_processor(lang, device):
    model_id = MODEL_MAPPING[lang]["name"]

    model = MODEL_MAPPING[lang].get("model", None)
    processor = MODEL_MAPPING[lang].get("processor", None)

    if processor == None:
        processor = WhisperProcessor.from_pretrained(model_id)
        MODEL_MAPPING[lang]["processor"] = processor

    if model == None:
        model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
        model = model.to(device)
        if device == "cuda":
            model.half()
        MODEL_MAPPING[lang]["model"] = model

    return model, processor


# copied from https://github.com/huggingface/transformers
def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        ) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError(
            "ffmpeg was not found but is required to load audio files from filename"
        ) from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


"""
VAD download and initialization
"""
print("Downloading VAD model")
model_vad, utils = silero_vad(True)

(get_speech_timestamps, _, read_audio, *_) = utils
