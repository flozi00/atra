import gradio as gr
from atra.gradio_utils.ui import GET_GLOBAL_HEADER, launch_args
from atra.text_utils.prompts import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    SYSTEM_PROMPT,
    SEARCH_PROMPT,
    USER_TOKEN,
    QUERY_PROMPT,
)
from atra.text_utils.assistant import Agent, Plugins
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os

embedder = SentenceTransformer("intfloat/multilingual-e5-large")

client = InferenceClient(model=os.environ.get("LLM", "http://127.0.0.1:8080"))

agent = Agent(client, embedder)


def get_user_messages(history, message):
    """
    Returns a string containing all the user messages in the chat history, including the current message.

    Args:
    - history (list): A list of tuples containing the user and their message.
    - message (str): The current message sent by the user.

    Returns:
    - A string containing all the user messages in the chat history, including the current message.
    """
    users = ""
    for h in history:
        users += USER_TOKEN + h[0] + END_TOKEN

    users += USER_TOKEN + message + END_TOKEN

    return users[-2048 * 3 :]


def generate_history_as_string(history, message):
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
                        USER_TOKEN + item[0] + END_TOKEN,
                        ASSISTANT_TOKEN + item[1] + END_TOKEN,
                    ]
                )
                for item in history
            ]
        )
    )

    messages += USER_TOKEN + message + END_TOKEN + ASSISTANT_TOKEN

    return messages


def predict(message, chatbot, url):
    yield "Reading History"
    input_prompt = generate_history_as_string(chatbot, message)
    user_messages = get_user_messages(chatbot, message)

    yield "Classifying Plugin"
    plugin = agent.classify_plugin(user_messages)

    if plugin == Plugins.SEARCH:
        yield "Generating Search Query"
        search_query = agent.generate_search_query(user_messages)
        if len(url) > 6:
            search_query += " site:" + url
        search_question = agent.generate_search_question(user_messages)
        
        yield "Searching"
        options = agent.get_webpage_content_playwright(search_query)

        yield "Answering"
        answer = agent.do_qa(search_question, options)
        for text in answer:
            yield text
    else:
        answer = agent.custom_generation(input_prompt)
        for text in answer:
            yield text


def build_chat_ui():
    with gr.Blocks() as demo:
        GET_GLOBAL_HEADER()
        gr.ChatInterface(predict, additional_inputs=[gr.Textbox(lines=1, label="Domain")])

    demo.queue(concurrency_count=4)
    demo.launch(server_port=7860, **launch_args)
