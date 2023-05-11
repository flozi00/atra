import hashlib
import os
from datetime import timedelta

import gradio as gr
import numpy as np
from transformers.pipelines.audio_utils import ffmpeg_read
from aaas.audio_utils.asr import inference_asr

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
from aaas.statics import LANG_MAPPING

langs = sorted(list(LANG_MAPPING.keys()))

model_vad, get_speech_timestamps = silero_vad(True)

is_admin_node = os.getenv("ADMINMODE", "false") == "true"


def build_edit_ui():
    def set_transkription(task_id, transcription):
        set_transkript(task_id, transcription)

    """
    UI for editing transcriptions and voting like confirm, good, bad
    """
    task_id = gr.Textbox(label="Task ID", max_lines=3)
    lang = gr.Radio(langs, value=langs[0], label="Source Language")

    audio_file = gr.Audio(label="Audiofile")
    transcription = gr.Textbox(max_lines=10)
    label = gr.Radio(["good", "bad", "confirm"], label="Label")
    with gr.Row():
        send = gr.Button(value="Send")
        if is_admin_node is True:
            rate = gr.Button(value="Rate")

    task_id.change(
        fn=get_audio,
        inputs=task_id,
        outputs=[audio_file, transcription],
        api_name="get_audio",
    )

    send.click(
        fn=set_transkription,
        inputs=[task_id, transcription],
        outputs=[],
        api_name="correct_transcription",
    )
    if is_admin_node is True:
        rate.click(
            fn=do_voting_labeling,
            inputs=[task_id, label, transcription, lang],
            outputs=[task_id],
        )


def build_asr_ui():
    """
    UI for ASR
    """
    with gr.Row():
        model_config = gr.Radio(
            choices=["small", "medium", "large"], value="large", label="model size"
        )

    with gr.Row():
        audio_file = gr.Audio(source="upload", type="filepath", label="Audiofile")

    task_id = gr.Textbox(label="Task ID", max_lines=3)

    audio_file.change(
        fn=add_to_vad_queue,
        inputs=[audio_file, model_config],
        outputs=[task_id],
        api_name="transcription",
    )


def build_realtime_asr_ui():
    """
    UI for ASR
    """
    with gr.Row():
        model_config = gr.Radio(
            choices=["small", "medium", "large"], value="large", label="model size"
        )

    with gr.Row():
        audio_file = gr.Audio(source="microphone", type="filepath", label="Audiofile")

    remove_audio = gr.Checkbox(label="Remove Audio", checked=False)

    transcript = gr.Textbox(label="Task ID", max_lines=3)
    lang = gr.Textbox(label="Language", max_lines=1)

    audio_file.change(
        fn=inference_asr,
        inputs=[audio_file, model_config, remove_audio],
        outputs=[transcript, lang],
        api_name="realtime_transcription",
    )


def build_results_ui():
    """
    UI for results page
    """
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
    """
    UI for voting page
    """
    task_id = gr.Textbox(label="Task ID", max_lines=3)

    with gr.Row():
        rating = gr.Radio(choices=["good", "bad"], value="good")

    task_id.change(
        fn=do_voting,
        inputs=[task_id, rating],
        outputs=[],
        api_name="vote_result",
    )


def build_subtitle_ui():
    task_id = gr.Textbox(label="Task ID", max_lines=3)
    srt_file = gr.File(label="SRT File")

    refresh = gr.Button(value="Get Results")

    refresh.click(
        fn=get_subs,
        inputs=[task_id],
        outputs=[srt_file],
        api_name="subtitle",
    )


def build_gradio():
    """
    Merge all UIs into one
    """
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
            with gr.Tab("Subtitles"):
                build_subtitle_ui()
            if is_admin_node is True:
                with gr.Tab("Realtime ASR"):
                    build_realtime_asr_ui()

    return ui


def do_voting(task_id: str, rating: str):
    if rating != "confirm":
        set_voting(task_id, rating)


def do_voting_labeling(task_id: str, rating: str, transcription: str, lang: str):
    if is_admin_node is True:
        set_transkript(task_id, transcription)
        set_voting(task_id, rating)
        queue_obj = get_vote_queue(lang)
        if queue_obj is False:
            return ""
        else:
            return queue_obj.hash


def add_to_vad_queue(audio: str, model_config: str):
    if model_config not in ["small", "medium", "large"]:
        model_config = "small"

    if audio is not None and len(audio) > 8:
        audio_path = audio

        with open(audio, "rb") as f:
            payload = f.read()

        audio = ffmpeg_read(payload, sampling_rate=16000)
        os.remove(audio_path)

    queue_string = add_vad_chunks(audio, model_config)

    return queue_string


def add_vad_chunks(audio, model_config: str):
    if model_config not in ["small", "medium", "large"]:
        model_config = "small"

    speech_timestamps = [{"start": 0, "end": len(audio) / 16000}]
    for silence_duration in [2000, 1000, 500, 100, 10]:
        temp_times = []
        for ts in speech_timestamps:
            if (ts["end"] - ts["start"]) > 20:
                temp_audio = audio[
                    int(float(ts["start"] * 16000)) : int(float(ts["end"] * 16000))
                ]
                speech_probs = get_speech_probs(
                    temp_audio,
                    model_vad,
                    sampling_rate=16000,
                )

                speech_timestamps_iteration = get_speech_timestamps(
                    temp_audio,
                    speech_probs=speech_probs,
                    threshold=0.6,
                    sampling_rate=16000,
                    min_silence_duration_ms=silence_duration,
                    min_speech_duration_ms=500,
                    speech_pad_ms=400,
                    return_seconds=True,
                )
                for temp_ts in speech_timestamps_iteration:
                    temp_times.append(
                        {
                            "start": ts["start"] + temp_ts["start"],
                            "end": ts["start"] + temp_ts["end"],
                        }
                    )

            else:
                temp_times += [ts]

        speech_timestamps = temp_times

    audio_batch = [
        audio[
            int(float(speech_timestamps[st]["start"]) * 16000) : int(
                float(speech_timestamps[st]["end"]) * 16000
            )
        ].tobytes()
        for st in range(len(speech_timestamps))
    ]

    file_format = "wav"
    hashes = []
    for x in range(len(audio_batch)):
        # Get the audio data
        audio_data = audio_batch[x]
        hs = hashlib.sha256(f"{audio_data} {model_config}".encode("utf-8")).hexdigest()
        # Add the file format to the hash
        hs = f"{hs}.{file_format}"
        # Add the hash to the list of hashes
        hashes.append(hs)

    add_to_queue(
        audio_batch=audio_batch,
        hashes=hashes,
        master=speech_timestamps,
        model_config=model_config,
    )

    queue_string = ",".join(hashes)

    return queue_string


def get_transcription(queue_string: str):
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


def get_audio(task_id: str):
    result = get_transkript(task_id)
    if result is None:
        return (16000, np.zeros(16000)), ""
    bytes_data = get_data_from_hash(result.hash)
    return (16000, np.frombuffer(bytes_data, dtype=np.float32)), result.transcript


def get_subs(task_id: str):
    segments = get_transcription(task_id)[1]
    srtFilename = hashlib.sha256(task_id.encode("utf-8")).hexdigest() + ".srt"
    if os.path.exists(srtFilename):
        os.remove(srtFilename)
    id = 1
    for segment in segments:
        startTime = (
            str(0) + str(timedelta(seconds=int(segment["start_timestamp"]))) + ",000"
        )
        endTime = (
            str(0) + str(timedelta(seconds=int(segment["stop_timestamp"]))) + ",000"
        )
        text = segment["text"]
        seg = f"{id}\n{startTime} --> {endTime}\n{text}\n\n"
        id = id + 1

        with open(srtFilename, "a+", encoding="utf-8") as srtFile:
            srtFile.write(seg)

    return srtFilename
