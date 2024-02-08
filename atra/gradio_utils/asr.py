import gradio as gr
from atra.audio_utils.asr import speech_recognition
from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from atra.audio_utils.whisper_langs import WHISPER_LANG_MAPPING

asr_langs = sorted(list(WHISPER_LANG_MAPPING.keys()))


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
            fn=speech_recognition,
            inputs=[microphone_file, input_lang],
            outputs=[transcription_finished],
        )

    ui.queue(api_open=False)

    return ui
