import os

import gradio as gr
from transformers.pipelines.audio_utils import ffmpeg_read

from aaas.audio_utils import LANG_MAPPING
from aaas.datastore import add_audio, delete_by_hashes, get_all_transkripts
from aaas.silero_vad import silero_vad
from aaas.statics import TODO
import time

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

        with gr.Tabs():
            with gr.TabItem("Transcription"):
                transcription = gr.Textbox()
            with gr.TabItem("details"):
                chunks = gr.JSON()

        mic.change(
            fn=run_transcription,
            inputs=[mic, lang, model_config, target_lang],
            outputs=[transcription, chunks],
            api_name="transcription",
        )
        audio_file.change(
            fn=run_transcription,
            inputs=[audio_file, lang, model_config, target_lang],
            outputs=[transcription, chunks],
        )

    return ui


def run_transcription(audio, main_lang, model_config, target_lang=""):
    queue = []
    chunks = []
    full_transcription = ""
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
            master=audio_path,
            main_lang=f"{main_lang},{target_lang}",
            model_config=model_config,
        )

        chunks = [
            {
                "id": queue[x],
                "text": TODO,
                "start_timestamp": (speech_timestamps[x]["start"] / 16000) - 0.1,
                "stop_timestamp": (speech_timestamps[x]["end"] / 16000) - 0.5,
            }
            for x in range(len(speech_timestamps))
        ]

        while TODO in str(chunks):
            results = get_all_transkripts()
            for x in range(len(queue)):
                if chunks[x]["text"] == TODO:
                    response = TODO
                    for res in results:
                        if res.hs == queue[x]:
                            response = res.transcript
                            chunks[x]["text"] = response

                            full_transcription = ""
                            for c in chunks:
                                full_transcription += c.get("text", "") + "\n"
                            yield full_transcription, chunks
                            delete_by_hashes([queue[x]])
            time.sleep(1)

    yield full_transcription, chunks
