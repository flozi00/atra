import base64
import gradio as gr
from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from atra.utilities.whisper_langs import WHISPER_LANG_MAPPING
from pytriton.client import ModelClient
import numpy as np

asr_langs = sorted(list(WHISPER_LANG_MAPPING.keys()))


def infer_client(audio: str, language: str) -> str:
    if audio is None:
        return ""
    with ModelClient("localhost", "Whisper") as client:
        with open(audio, "rb") as f:
            audio = f.read()
        audio = base64.b64encode(audio)
        audio = np.array([[audio]])
        language = np.array([[language]])
        language = np.char.encode(language, "utf-8")

        result_dict = client.infer_batch(language=language, audio=audio)

        transcription = result_dict["transcription"]
        transcription = [
            np.char.decode(p.astype("bytes"), "utf-8").item() for p in transcription
        ]

    return transcription[0]


def build_asr_ui():
    ui = gr.Blocks(css=GLOBAL_CSS)
    with ui:
        GET_GLOBAL_HEADER()
        input_lang = gr.Dropdown(
            choices=asr_langs, value="german", label="Input Language"
        )
        with gr.Row():
            with gr.TabItem("Microphone"):
                microphone_file = gr.Audio(type="filepath", label="Audio")

        with gr.Row():
            with gr.TabItem("Transcription"):
                transcription_finished = gr.Textbox(max_lines=10)

        microphone_file.change(
            fn=infer_client,
            inputs=[microphone_file, input_lang],
            outputs=[transcription_finished],
        )

    ui.queue(api_open=False)

    ui.launch(**launch_args)
