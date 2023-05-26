import gradio as gr
from atra.audio_utils.asr import speech_recognition

from atra.statics import LANG_MAPPING, MODEL_MAPPING
from atra.text_utils.question_answering import answer_question
from atra.text_utils.summarization import summarize
from atra.text_utils.translation import translate

langs = sorted(list(LANG_MAPPING.keys()))
sum_langs = sorted(list(MODEL_MAPPING["summarization"].keys()))
question_answering_langs = sorted(list(MODEL_MAPPING["question-answering"].keys()))

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
    input_lang = gr.Dropdown(langs, value="german", label="Input Language")
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
            input_lang = gr.Dropdown(langs, value=langs[0], label="Input Language")
            input_text = gr.Textbox(label="Input Text")

        with gr.Column():
            output_lang = gr.Dropdown(langs, value=langs[0], label="Output Language")
            output_text = gr.Text(label="Output Text")

    send = gr.Button(label="Translate")

    send.click(
        translate,
        inputs=[input_text, input_lang, output_lang],
        outputs=[output_text],
    )


def build_summarization_ui():
    with gr.Row():
        with gr.Column():
            input_lang = gr.Dropdown(
                sum_langs, value="german", label="Input Language"
            )
            input_text = gr.Textbox(label="Input Text")

        output_text = gr.Text(label="Summarization")

    send = gr.Button(label="Summarize")

    send.click(
        summarize,
        inputs=[input_text, input_lang],
        outputs=[output_text],
    )


def build_question_answering_ui():
    with gr.Row():
        with gr.Column():
            input_lang = gr.Dropdown(
                question_answering_langs,
                value="german",
                label="Input Language",
            )
            with gr.Row():
                input_text = gr.Textbox(label="Input Text")
                question = gr.Textbox(label="Question")

        output_text = gr.Text(label="Answer")

    send = gr.Button(label="Answer")

    send.click(
        answer_question,
        inputs=[input_text, question, input_lang],
        outputs=[output_text],
    )
