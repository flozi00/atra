import hashlib
import os
from atra.datastore import add_to_queue, delete_by_hash
from transformers.pipelines.audio_utils import ffmpeg_read

from atra.gradio_utils.utils import get_transcription


def add_to_translation_queue(input_text: str, input_lang: str, output_lang: str):
    input_text_bytes = input_text.encode(encoding="UTF-8")
    hs = hashlib.sha256(input_text_bytes).hexdigest()
    hs = f"{hs}.txt"

    add_to_queue(
        audio_batch=[input_text_bytes],
        hashes=[hs],
        times_list=[{"source": input_lang, "target": output_lang}],
    )

    transcript, chunks = get_transcription(hs)
    while "***" in transcript:
        transcript, chunks = get_transcription(hs)

    delete_by_hash(hs)

    return transcript


def add_to_summarization_queue(input_text: str, input_lang: str):
    input_text_bytes = input_text.encode(encoding="UTF-8")
    hs = hashlib.sha256(input_text_bytes).hexdigest()
    hs = f"{hs}.txt"

    add_to_queue(
        audio_batch=[input_text_bytes],
        hashes=[hs],
        times_list=[{"long_text": input_lang, "short_text": input_lang}],
    )

    transcript, chunks = get_transcription(hs)
    while "***" in transcript:
        transcript, chunks = get_transcription(hs)

    delete_by_hash(hs)

    return transcript


def add_to_vad_queue(audio: str, lang: str):
    hs = ""
    if audio is not None and len(audio) > 8:
        audio_path = audio

        with open(audio, "rb") as f:
            payload = f.read()

        audio = ffmpeg_read(payload, sampling_rate=16000)
        os.remove(audio_path)

        speech_timestamps = [{"start": 0, "end": len(audio) / 16000, "language" : lang}]

        file_format = "wav"
        hs = hashlib.sha256(f"{audio}".encode("utf-8")).hexdigest()
        hs = f"{hs}.{file_format}"

        add_to_queue(
            audio_batch=[audio.tobytes()], # type: ignore
            hashes=[hs],
            times_list=speech_timestamps,
        )

    return hs
