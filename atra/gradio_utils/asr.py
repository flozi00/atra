import gradio as gr
from atra.audio_utils.asr import speech_recognition
from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from atra.statics import WHISPER_LANG_MAPPING

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
                microphone_file = gr.Audio(
                    source="microphone", type="filepath", label="Audio"
                )
            with gr.TabItem("File Upload"):
                audio_file = gr.Audio(
                    source="upload", type="filepath", label="Audiofile"
                )

        with gr.Row():
            with gr.TabItem("Transcription"):
                transcription_finished = gr.Textbox(max_lines=10)

        audio_file.change(
            fn=speech_recognition,
            inputs=[audio_file, input_lang],
            outputs=[transcription_finished],
            api_name="transcription",
        )
        microphone_file.change(
            fn=speech_recognition,
            inputs=[microphone_file, input_lang],
            outputs=[transcription_finished],
        )

    ui.queue(concurrency_count=1, api_open=False)
    ui.launch(server_port=7862, **launch_args)