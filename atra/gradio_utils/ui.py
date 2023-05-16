import gradio as gr

from atra.gradio_utils.queues import (
    add_to_summarization_queue,
    add_to_translation_queue,
    add_to_vad_queue,
)
from atra.gradio_utils.utils import get_transcription, wait_for_transcription
from atra.statics import LANG_MAPPING, MODEL_MAPPING

langs = sorted(list(LANG_MAPPING.keys()))
sum_langs = sorted(list(MODEL_MAPPING["summarization"].keys()))

GLOBAL_CSS = """
#hidden_stuff {display: none} 
"""


def build_asr_ui():
    """
    UI for ASR
    """
    # UI for getting audio
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
        with gr.TabItem("details"):
            chunks_finished = gr.JSON()

    # hidden UI stuff
    with gr.Row(elem_id="hidden_stuff"):
        task_id = gr.Textbox(label="Task ID", max_lines=3)
        with gr.TabItem("Transcription State"):
            with gr.Row():
                with gr.TabItem("Transcription"):
                    transcription = gr.Textbox(max_lines=10)
                with gr.TabItem("details"):
                    chunks = gr.JSON()

    audio_file.change(
        fn=add_to_vad_queue,
        inputs=[audio_file],
        outputs=[task_id],
        api_name="transcription",
    )
    microphone_file.change(
        fn=add_to_vad_queue,
        inputs=[microphone_file],
        outputs=[task_id],
    )

    task_id.change(
        fn=get_transcription,
        inputs=task_id,
        outputs=[transcription, chunks],
        api_name="get_transcription",
    )

    task_id.change(
        fn=wait_for_transcription,
        inputs=task_id,
        outputs=[transcription_finished, chunks_finished],
    )


def build_translator_ui():
    with gr.Row():
        with gr.Column():
            input_lang = gr.Dropdown(langs)
            input_text = gr.Textbox(label="Input Text")

        with gr.Column():
            output_lang = gr.Dropdown(langs)
            output_text = gr.Text(label="Output Text")

    send = gr.Button(label="Translate")

    send.click(
        add_to_translation_queue,
        inputs=[input_text, input_lang, output_lang],
        outputs=[output_text],
    )


def build_summarization_ui():
    with gr.Row():
        with gr.Column():
            input_lang = gr.Dropdown(sum_langs)
            input_text = gr.Textbox(label="Input Text")

        output_text = gr.Text(label="Output Text")

    send = gr.Button(label="Translate")

    send.click(
        add_to_summarization_queue,
        inputs=[input_text, input_lang],
        outputs=[output_text],
    )
