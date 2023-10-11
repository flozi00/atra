import gradio as gr
from atra.gradio_utils.ui import GET_GLOBAL_HEADER, launch_args
from atra.text_utils.prompts import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    SYSTEM_PROMPT,
    USER_TOKEN,
)
from atra.text_utils.assistant import Agent, Plugins
from huggingface_hub import InferenceClient
import os
import csv
from atra.text_utils.typesense_search import Embedder

embedder = Embedder("intfloat/multilingual-e5-large")

client = InferenceClient(model=os.environ.get("LLM", "http://127.0.0.1:8080"))

agent = Agent(client, embedder)


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
    history_no_tokens = (
        input_prompt.rstrip(ASSISTANT_TOKEN).rstrip().replace(SYSTEM_PROMPT, "").strip()
    )

    yield "Classifying Plugin"
    if input_prompt.count(USER_TOKEN) == 1:
        search_question = message
    else:
        search_question = agent.generate_selfstanding_query(history_no_tokens)
    plugin = agent.classify_plugin(search_question)

    if plugin == Plugins.SEARCH:
        yield "Suche: " + search_question
        if os.getenv("TYPESENSE_API_KEY") is None:
            search_query = search_question
            if len(url) > 6:
                search_query += f" site:{url}"
            options = agent.get_webpage_content_playwright(search_query)
        else:
            options = agent.get_data_from_typesense(search_question)

        yield "Answering"
        answer = agent.do_qa(search_question, options)
        for text in answer:
            yield text
    else:
        answer = agent.custom_generation(input_prompt)
        for text in answer:
            yield text


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

    with open("chat-feedback.csv", mode="a+") as file:
        writer = csv.writer(file)
        writer.writerow([chat, "Liked" if evt.liked else "Disliked"])

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

        chatter.like(on_like, chatter)

    demo.queue(concurrency_count=4)
    demo.launch(**launch_args)
