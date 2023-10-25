import gradio as gr
from atra.gradio_utils.ui import GET_GLOBAL_HEADER, launch_args
from atra.text_utils.prompts import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    SYSTEM_PROMPT,
    USER_TOKEN,
)
from atra.text_utils.assistant import Agent
from huggingface_hub import InferenceClient
import os
from atra.text_utils.typesense_search import Embedder

embedder = Embedder()

client = InferenceClient(model=os.getenv("LLM", "http://127.0.0.1:8080"))

agent = Agent(client, embedder, creative=True)


def generate_history_as_string(history: list, message: str) -> str:
    """
    Generates a string representation of the chat history and the current message.

    Args:
        history (list): A list of tuples containing the user and assistant messages.
        message (str): The current message to be added to the chat history.

    Returns:
        str: A string representation of the chat history and the current message.
    """
    messages = (
        SYSTEM_PROMPT
        + "\n\n"
        + "\n".join(
            [
                "\n".join(
                    [
                        USER_TOKEN + item[0].rstrip() + END_TOKEN,
                        ASSISTANT_TOKEN + item[1].rstrip() + END_TOKEN,
                    ]
                )
                for item in history
            ]
        )
    )

    messages += "\n" + USER_TOKEN + message.rstrip() + END_TOKEN + ASSISTANT_TOKEN

    return messages[-4096 * 3 :].strip()


def predict(message: str, chatbot: list, url: str):
    yield "Reading History"
    input_prompt = generate_history_as_string(chatbot, message)

    yield "Generating Response"
    response = agent.__call__(last_message=message, full_history=input_prompt, url=url)

    for r in response:
        yield r


def label_chat(history: list, message: str) -> str:
    messages = "\n".join(
        [
            "\n".join(
                [
                    USER_TOKEN + item[0].rstrip() + END_TOKEN,
                    ASSISTANT_TOKEN + item[1].rstrip() + END_TOKEN,
                ]
            )
            for item in history
        ]
    )

    messages = messages.split(message)[0] + message + END_TOKEN

    return messages


def on_like(evt: gr.LikeData, history: list) -> None:
    chat = label_chat(history, evt.value)

    agent.log_text2text(
        input=chat, output="Liked" if evt.liked else "Disliked", tasktype="Feedback"
    )

    return gr.Info("Thanks for your feedback!")


chatter = gr.Chatbot()


def build_chat_ui():
    with gr.Blocks() as demo:
        GET_GLOBAL_HEADER()
        gr.ChatInterface(
            predict,
            chatbot=chatter,
            additional_inputs=[gr.Textbox(lines=1, label="Domain")],
        )

        # chatter.like(on_like, chatter)

    demo.queue()
    demo.launch(**launch_args)
