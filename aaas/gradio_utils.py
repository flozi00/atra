import os

import gradio as gr
from transformers.pipelines.audio_utils import ffmpeg_read

from aaas.statics import LANG_MAPPING
from aaas.datastore import add_audio, get_transkript
from aaas.silero_vad import silero_vad

langs = sorted(list(LANG_MAPPING.keys()))

model_vad, get_speech_timestamps = silero_vad(True)


def build_vad():
    with gr.Row():
        audio_file = gr.Audio(source="upload", type="filepath", label="Audiofile")

    with gr.Row():
        chunks = gr.JSON()

    audio_file.change(
        fn=run_vad, inputs=[audio_file], outputs=[chunks], api_name="vad",
    )


def build_asr_ui(lang, model_config, target_lang):
    with gr.Row():
        audio_file = gr.Audio(source="upload", type="filepath", label="Audiofile")

    task_id = gr.Textbox(label="Task ID", max_lines=3)

    with gr.Row():
        with gr.TabItem("Transcription"):
            transcription = gr.Textbox(max_lines=10)
        with gr.TabItem("details"):
            chunks = gr.JSON()

    refresh = gr.Button(value="Get Results")

    audio_file.change(
        fn=run_transcription,
        inputs=[audio_file, lang, model_config, target_lang],
        outputs=[task_id],
        api_name="transcription",
    )

    task_id.change(
        fn=get_transcription,
        inputs=task_id,
        outputs=[transcription, chunks],
        api_name="get_transcription",
    )

    refresh.click(fn=get_transcription, inputs=task_id, outputs=[transcription, chunks])


def build_gradio():
    ui = gr.Blocks()

    with ui:
        with gr.Row():
            lang = gr.Radio(langs, value=langs[0], label="Source Language")
            model_config = gr.Radio(
                choices=["small", "medium", "large"], value="large", label="model size"
            )
            target_lang = gr.Radio(langs, label="Target language")
        with gr.Tabs():
            with gr.Tab("ASR"):
                build_asr_ui(lang, model_config, target_lang)
            with gr.Tab("VAD"):
                build_vad()

    return ui


def run_vad(audio):
    speech_timestamps = []
    if audio is not None and len(audio) > 3:

        if isinstance(audio, str):
            with open(audio, "rb") as f:
                payload = f.read()

            audio = ffmpeg_read(payload, sampling_rate=16000)

        speech_timestamps = get_speech_timestamps(
            audio,
            model_vad,
            threshold=0.5,
            sampling_rate=16000,
            min_silence_duration_ms=250,
            min_speech_duration_ms=1000,
            speech_pad_ms=100,
            return_seconds=True,
        )

    return speech_timestamps


def run_transcription(audio, main_lang, model_config, target_lang=""):
    queue_string = ""
    if main_lang not in langs:
        main_lang = "german"
        target_lang = "german"
    if model_config not in ["small", "medium", "large"]:
        model_config = "small"

    queue = []
    if target_lang not in langs:
        target_lang = main_lang

    if audio is not None and len(audio) > 3:
        audio_path = audio

        with open(audio, "rb") as f:
            payload = f.read()

        audio = ffmpeg_read(payload, sampling_rate=16000)
        os.remove(audio_path)

        speech_timestamps = run_vad(audio)
        audio_batch = [
            audio[
                int(float(speech_timestamps[st]["start"]) * 16000) : int(
                    float(speech_timestamps[st]["end"]) * 16000
                )
            ]
            for st in range(len(speech_timestamps))
        ]

        queue = add_audio(
            audio_batch=audio_batch,
            master=speech_timestamps,
            main_lang=f"{main_lang},{target_lang}",
            model_config=model_config,
        )

        queue_string = ",".join(queue)

    return queue_string


def get_transcription(queue_string: str):
    queue_string = str(queue_string)
    if len(queue_string) < 5:
        return "", []

    full_transcription = ""
    queue = queue_string.split(",")

    chunks = [{"id": queue[x]} for x in range(len(queue))]

    for x in range(len(queue)):
        result = get_transkript(queue[x])
        if result is not None:
            chunks[x]["start_timestamp"] = int(float(result.timestamps.split(",")[0]))
            chunks[x]["stop_timestamp"] = int(float(result.timestamps.split(",")[1]))
            chunks[x]["text"] = result.transcript

        full_transcription = ""
        for c in chunks:
            full_transcription += c.get("text", "") + "\n"

    return full_transcription, chunks
