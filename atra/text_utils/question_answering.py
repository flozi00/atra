from atra.model_utils.model_utils import get_prompt
from atra.text_utils.generations import do_generation
import gradio as gr


def answer_question(text, question, input_lang, progress=gr.Progress()) -> str:
    progress.__call__(0.2, "Filtering Text")
    text = get_prompt(task="question-answering", lang=input_lang).format(
        text=text, question=question
    )
    progress.__call__(0.8, "Answering Question")
    generated_tokens = do_generation(text)
    return generated_tokens