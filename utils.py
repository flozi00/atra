import subprocess
import numpy as np
from silero_vad import silero_vad

MODEL_MAPPING = {
    "german": "aware-ai/wav2vec2-xls-r-300m-german",
    "english": "aware-ai/wav2vec2-xls-r-300m-english",
    "german-english": "aware-ai/wav2vec2-xls-r-300m-german-english",
}

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
