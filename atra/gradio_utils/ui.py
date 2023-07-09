import gradio as gr
from atra.audio_utils.asr import speech_recognition
from atra.image_utils.diffusion import generate_images

from atra.statics import WHISPER_LANG_MAPPING

asr_langs = sorted(list(WHISPER_LANG_MAPPING.keys()))

GLOBAL_CSS = """
#hidden_stuff {display: none} 
"""


def build_asr_ui():
    """
    UI for ASR
    It has 2 tabs for getting audio:
    1. Microphone
    2. File Upload

    It has 2 tabs for showing the results:
    1. Transcription
    2. Details in JSON format
    """
    # UI for getting audio
    input_lang = gr.Dropdown(choices=asr_langs, value="german", label="Input Language")
    with gr.Row():
        with gr.TabItem("Microphone"):
            microphone_file = gr.Audio(
                source="microphone", type="filepath", label="Audio"
            )
        with gr.TabItem("File Upload"):
            audio_file = gr.Audio(source="upload", type="filepath", label="Audiofile")

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


def build_diffusion_ui():
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            negatives = gr.Textbox(label="Negative Prompt")
        images = gr.Image()

    prompt.submit(generate_images, inputs=[prompt, negatives], outputs=images)
