import hashlib
import os
from atra.datastore import add_to_queue, delete_by_hash
from transformers.pipelines.audio_utils import ffmpeg_read

from atra.gradio_utils.utils import add_vad_chunks, get_transcription


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


def add_to_vad_queue(audio: str):
    if audio is not None and len(audio) > 8:
        audio_path = audio

        with open(audio, "rb") as f:
            payload = f.read()

        audio = ffmpeg_read(payload, sampling_rate=16000)
        os.remove(audio_path)

        queue_string = add_vad_chunks(audio)
    else:
        queue_string = ""

    return queue_string
