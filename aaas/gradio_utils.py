import os

import gradio as gr
from transformers.pipelines.audio_utils import ffmpeg_read

from aaas.statics import LANG_MAPPING
from aaas.datastore import add_audio, get_transkript
from aaas.silero_vad import silero_vad

langs = sorted(list(LANG_MAPPING.keys()))

model_vad, get_speech_timestamps = silero_vad(True)


def build_gradio():
    ui = gr.Blocks()

    with ui:
        with gr.Tabs():
            with gr.TabItem("audio language"):
                lang = gr.Radio(langs, value=langs[0])
            with gr.TabItem("model configuration"):
                model_config = gr.Radio(
                    choices=["small", "medium", "large"], value="large"
                )
            with gr.TabItem("translate to"):
                target_lang = gr.Radio(langs)

        with gr.Tabs():
            with gr.TabItem("Microphone"):
                mic = gr.Audio(source="microphone", type="filepath")
            with gr.TabItem("File"):
                audio_file = gr.Audio(source="upload", type="filepath")

        task_id = gr.Textbox(label="Task ID")
        refresh = gr.Button(value="Get Results")

        with gr.Tabs():
            with gr.TabItem("Transcription"):
                transcription = gr.Textbox()
            with gr.TabItem("details"):
                chunks = gr.JSON()

        mic.change(
            fn=run_transcription,
            inputs=[mic, lang, model_config, target_lang],
            outputs=[task_id],
            api_name="transcription",
        )
        audio_file.change(
            fn=run_transcription,
            inputs=[audio_file, lang, model_config, target_lang],
            outputs=[task_id],
        )

        task_id.change(
            fn=get_transcription, inputs=task_id, outputs=[transcription, chunks]
        )

        refresh.click(
            fn=get_transcription, inputs=task_id, outputs=[transcription, chunks]
        )

    return ui


def run_transcription(audio, main_lang, model_config, target_lang=""):
    if main_lang not in langs:
        main_lang = "german"
        target_lang = "german"
    if model_config not in ["small", "medium", "large"]:
        model_config = "small"

    queue = []
    if target_lang == "":
        target_lang = main_lang

    if audio is not None and len(audio) > 3:
        audio_path = audio

        with open(audio, "rb") as f:
            payload = f.read()

        audio = ffmpeg_read(payload, sampling_rate=16000)
        os.remove(audio_path)

        if len(audio) > 29 * 16000:
            speech_timestamps = get_speech_timestamps(
                audio,
                model_vad,
                threshold=0.5,
                sampling_rate=16000,
                min_silence_duration_ms=250,
                speech_pad_ms=100,
            )
            audio_batch = [
                audio[speech_timestamps[st]["start"] : speech_timestamps[st]["end"]]
                for st in range(len(speech_timestamps))
            ]
        else:
            speech_timestamps = [{"start": 100, "end": len(audio)}]
            audio_batch = [audio]

        queue = add_audio(
            audio_batch=audio_batch,
            master=speech_timestamps,
            main_lang=f"{main_lang},{target_lang}",
            model_config=model_config,
        )

        queue_string = ",".join(queue)

        return queue_string


def get_transcription(queue_string: str):
    if len(queue_string) < 5:
        return "", []

    full_transcription = ""
    queue = queue_string.split(",")

    chunks = [{"id": queue[x]} for x in range(len(queue))]

    for x in range(len(queue)):
        result = get_transkript(queue[x])
        if result is not None:
            chunks[x]["start_timestamp"] = int(result.master.split(",")[0]) / 16000
            chunks[x]["stoip_timestamp"] = int(result.master.split(",")[1]) / 16000
            chunks[x]["text"] = result.transcript

        full_transcription = ""
        for c in chunks:
            full_transcription += c.get("text", "") + "\n"

    return full_transcription, chunks
