import gradio as gr
from atra.gradio_utils.assistant import build_chatbot_ui
from atra.gradio_utils.ui import (
    GLOBAL_CSS,
    build_asr_ui,
    build_summarization_ui,
    build_translator_ui,
    build_question_answering_ui,
)


def build_gradio():
    """
    Merge all UIs into one
    """
    ui = gr.Blocks(css=GLOBAL_CSS)

    with ui:
        with gr.Row():
            gr.Markdown('## This project is sponsored by ')
            gr.Markdown('[ ![PrimeLine](https://www.primeline-solutions.com/skin/frontend/default/theme566/images/primeline-solutions-logo.png) ](https://www.primeline-solutions.com/de/server/nach-einsatzzweck/gpu-rendering-hpc/)')
        with gr.Tabs():
            with gr.Tab("Chat"):
                build_chatbot_ui()
            with gr.Tab("Tools"):
                with gr.Tabs():
                    with gr.Tab("ASR"):
                        build_asr_ui()
                    with gr.Tab("Translator"):
                        build_translator_ui()
                    with gr.Tab("Summarization"):
                        build_summarization_ui()
                    with gr.Tab("Question Answering"):
                        build_question_answering_ui()

    return ui
