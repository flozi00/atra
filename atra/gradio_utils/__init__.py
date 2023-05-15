import gradio as gr
from atra.gradio_utils.ui import (
    GLOBAL_CSS,
    build_asr_ui,
    build_summarization_ui,
    build_translator_ui,
)


def build_gradio():
    """
    Merge all UIs into one
    """
    ui = gr.Blocks(css=GLOBAL_CSS)

    with ui:
        with gr.Tabs():
            with gr.Tab("ASR"):
                build_asr_ui()
            with gr.Tab("Translator"):
                build_translator_ui()
            with gr.Tab("Summarization"):
                build_summarization_ui()

    return ui
