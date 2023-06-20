import gradio as gr
from atra.audio_utils.asr import speech_recognition
from atra.image_utils.diffusion import generate_images

from atra.statics import WHISPER_LANG_MAPPING, FLORES_LANG_MAPPING
from atra.text_utils.translation import translate

asr_langs = sorted(list(WHISPER_LANG_MAPPING.keys()))
translation_langs = sorted(list(FLORES_LANG_MAPPING.keys()))

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


def build_translator_ui():
    """
    UI for Translation
    It has one row with two columns
    Left side is for input
    Right side is for output
    """
    with gr.Row():
        with gr.Column():
            input_lang = gr.Dropdown(
                choices=translation_langs,
                value=translation_langs[0],
                label="Input Language",
            )
            input_text = gr.Textbox(label="Input Text")

        with gr.Column():
            output_lang = gr.Dropdown(
                choices=translation_langs,
                value=translation_langs[0],
                label="Output Language",
            )
            output_text = gr.Text(label="Output Text")

    input_text.submit(
        translate,
        inputs=[input_text, input_lang, output_lang],
        outputs=[output_text],
    )


def build_diffusion_ui():
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            num_steps = gr.Slider(minimum=5, maximum=50, value=20)
        images = gr.Image()

    prompt.submit(generate_images, inputs=[prompt, num_steps], outputs=images)
