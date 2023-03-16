from datetime import timedelta
import os

import gradio as gr
from transformers.pipelines.audio_utils import ffmpeg_read

from aaas.statics import LANG_MAPPING
from aaas.datastore import (
    add_to_queue,
    get_data_from_hash,
    get_transkript,
    get_transkript_batch,
    get_vote_queue,
    set_transkript,
    set_voting,
)
from aaas.silero_vad import get_speech_probs, silero_vad
import numpy as np

from aaas.utils import check_valid_auth

langs = sorted(list(LANG_MAPPING.keys()))

model_vad, get_speech_timestamps = silero_vad(True)


def build_edit_ui():
    task_id = gr.Textbox(label="Task ID", max_lines=3)

    audio_file = gr.Audio(label="Audiofile")
    transcription = gr.Textbox(max_lines=10)
    label = gr.Radio(["good", "bad", "confirm"], label="Label")
    with gr.Row():
        send = gr.Button(value="Send")
        rate = gr.Button(value="Rate")

    task_id.change(
        fn=get_audio,
        inputs=task_id,
        outputs=[audio_file, transcription],
        api_name="get_audio",
    )

    send.click(
        fn=set_transkript,
        inputs=[task_id, transcription],
        outputs=[],
        api_name="correct_transcription",
    )
    rate.click(fn=do_voting, inputs=[task_id, label], outputs=[task_id])


def build_asr_ui():
    with gr.Row():
        lang = gr.Radio(langs, value=langs[0], label="Source Language")
        model_config = gr.Radio(
            choices=["small", "medium", "large"], value="large", label="model size"
        )

    with gr.Row():
        audio_file = gr.Audio(source="upload", type="filepath", label="Audiofile")

    task_id = gr.Textbox(label="Task ID", max_lines=3)

    audio_file.change(
        fn=add_to_vad_queue,
        inputs=[audio_file, lang, model_config],
        outputs=[task_id],
        api_name="transcription",
    )


def build_results_ui():
    task_id = gr.Textbox(label="Task ID", max_lines=3)

    with gr.Row():
        with gr.TabItem("Transcription"):
            transcription = gr.Textbox(max_lines=10)
        with gr.TabItem("details"):
            chunks = gr.JSON()

    token = gr.Textbox(label="Token", max_lines=1, visible=False)

    task_id.change(
        fn=get_transcription,
        inputs=task_id,
        outputs=[transcription, chunks, token],
        api_name="get_transcription",
    )


def build_voting_ui():
    task_id = gr.Textbox(label="Task ID", max_lines=3)

    with gr.Row():
        rating = gr.Radio(choices=["good", "bad"], value="good")

    task_id.change(
        fn=do_voting,
        inputs=[task_id, rating],
        outputs=[],
        api_name="vote_result",
    )


def build_gradio():
    ui = gr.Blocks()

    with ui:
        with gr.Tabs():
            with gr.Tab("ASR"):
                build_asr_ui()
            with gr.Tab("Results"):
                build_results_ui()
            with gr.Tab("Edit"):
                build_edit_ui()
            with gr.Tab("Voting"):
                build_voting_ui()

    return ui


def do_voting(task_id, rating, request: gr.Request):
    if request.client.host != "127.0.0.1" and rating != "confirm":
        set_voting(task_id, rating)
        return task_id
    elif request.client.host == "127.0.0.1":
        set_voting(task_id, rating)
        return get_vote_queue().hash


def add_to_vad_queue(audio, main_lang, model_config, request: gr.Request):
    if main_lang not in langs:
        main_lang = "german"
    if model_config not in ["small", "medium", "large"]:
        model_config = "small"

    if audio is not None and len(audio) > 8:
        audio_path = audio

        with open(audio, "rb") as f:
            payload = f.read()

        audio = ffmpeg_read(payload, sampling_rate=16000)
        os.remove(audio_path)

    queue_string = add_vad_chunks(audio, main_lang, model_config, request=request)

    return queue_string


def add_vad_chunks(audio, main_lang, model_config, request: gr.Request):
    def check_timestamp_length_limit(timestamps, limit):
        for timestamp in timestamps:
            if timestamp["start"] - timestamp["end"] > limit:
                return True
        return False

    queue_string = ""
    speech_timestamps = []
    silence_duration = 750
    if main_lang not in langs:
        main_lang = "german"
    if model_config not in ["small", "medium", "large"]:
        model_config = "small"

    queue = []
    # audio = seperate_vocal(audio)

    speech_probs = get_speech_probs(
        audio,
        model_vad,
        sampling_rate=16000,
    )
    while len(speech_timestamps) < 1 or check_timestamp_length_limit(
        speech_timestamps, 20
    ):
        speech_timestamps = get_speech_timestamps(
            audio,
            model_vad,
            speech_probs=speech_probs,
            threshold=0.4,
            sampling_rate=16000,
            min_silence_duration_ms=silence_duration,
            min_speech_duration_ms=1000,
            speech_pad_ms=500,
            return_seconds=True,
        )
        silence_duration = silence_duration * 0.9
        if silence_duration < 2:
            break
    audio_batch = [
        audio[
            int(float(speech_timestamps[st]["start"]) * 16000) : int(
                float(speech_timestamps[st]["end"]) * 16000
            )
        ].tobytes()
        for st in range(len(speech_timestamps))
    ]

    print(len(audio_batch), len(audio) / 16000)
    queue = add_to_queue(
        audio_batch=audio_batch,
        master=speech_timestamps,
        main_lang=f"{main_lang}",
        model_config=model_config,
    )

    queue_string = ",".join(queue)

    return queue_string


def get_transcription(queue_string: str, request: gr.Request):
    if check_valid_auth(request=request) is not False:
        tok = "Valid"
    else:
        tok = "Invalid"
    full_transcription, chunks = "", []
    queue = queue_string.split(",")
    results = get_transkript_batch(queue_string)
    for x in range(len(results)):
        result = results[x]
        if result is None:
            return "", [], tok
        else:
            if len(queue_string) < 5 or "***" in queue_string:
                return queue_string, [], tok

            chunks.append({"id": queue[x]})
            if result is not None:
                try:
                    chunks[x]["start_timestamp"] = int(
                        float(result.metas.split(",")[0])
                    )
                    chunks[x]["stop_timestamp"] = int(float(result.metas.split(",")[1]))
                except Exception as e:
                    print(e)
                chunks[x]["text"] = result.transcript

    chunks = sorted(chunks, key=lambda d: d["start_timestamp"])

    full_transcription = ""
    for c in chunks:
        full_transcription += c.get("text", "") + "\n"

    return full_transcription, chunks, tok


def get_audio(task_id, request: gr.Request):
    result = get_transkript(task_id)
    bytes_data = get_data_from_hash(result.hash)
    return (16000, np.frombuffer(bytes_data, dtype=np.float32)), result.transcript
