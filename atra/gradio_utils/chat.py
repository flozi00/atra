import gradio as gr
from atra.gradio_utils.ui import GET_GLOBAL_HEADER, launch_args
from playwright.sync_api import sync_playwright
import urllib.parse
from atra.text_utils.prompts import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    SYSTEM_PROMPT,
    SEARCH_PROMPT,
    CLASSIFY_SEARCHABLE,
    USER_TOKEN,
    QUERY_PROMPT,
)
from sentence_transformers import SentenceTransformer, util
import torch
from huggingface_hub import InferenceClient
import os

embedder = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

client = InferenceClient(model=os.environ.get("LLM", "http://127.0.0.1:8080"))

def re_ranking(query, options):
    """
    1. Embeds the query and the corpus using the embedding model
    2. Calculates the cosine similarity between the query embedding and all corpus embeddings
    3. Filters the corpus based on the cosine similarity
    4. Returns the filtered corpus
    """
    corpus = ["passage: " + o for o in options]
    query = "query: " + query

    filtered_corpus = []

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=20 if len(corpus) > 20 else len(corpus))

    for score, idx in zip(top_results[0], top_results[1]):
        if score > 0.7:
            filtered_corpus.append(corpus[idx])

    return "\n".join(filtered_corpus)


def get_webpage_content_playwright(query):
    """
    1. It starts a chromium browser
    2. It goes to the URL with the query
    3. It gets the content of the body
    4. It splits the content at every new line
    5. It filters out all lines with less than 5 words
    6. It re-ranks the filtered lines
    7. It returns the re-ranked lines
    """
    url = (
        "https://searx.be/search?categories=general&language=de&q="
        + urllib.parse.quote(query)
    )
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.locator("body").inner_text()
        browser.close()

    content = content.split("\n")
    filtered = ""
    for co in content:
        if len(co.split(" ")) > 5:
            filtered += co + "\n"

    filtered = re_ranking(query, filtered.split("\n"))

    return filtered


def get_user_messages(history, message):
    users = ""
    for h in history:
        users += USER_TOKEN + h[0] + END_TOKEN

    users += USER_TOKEN + message + END_TOKEN

    return users[-2048*3:]


def generate_history_as_string(history, message):
    """
    1. Creates a string containing the history and the current input message.
    2. Creates a list containing the user and assistant messages.
    3. Creates a string containing the history and the current input message.
    4. Returns the string.
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


def predict(message, chatbot):
    """
    1. Prepare the input prompt for the model.
    2. Generate a response with the model.
    3. Check whether the response is searchable or not.
    4. If the response is searchable, generate a search query and search question.
    5. Get the answer from the internet.
    6. If the response is not searchable, generate a new response.
    7. If the response is not searchable, yield the response.
    8. If the response is searchable, yield the answer from the internet.
    """
    input_prompt = generate_history_as_string(chatbot, message)
    user_messages = get_user_messages(chatbot, message)
    searchable_answer = client.text_generation(prompt=
        CLASSIFY_SEARCHABLE.replace("<|question|>", user_messages),
        temperature=0.1,
        stop_sequences=["\n"],
        max_new_tokens=3,
    )
    searchable = "Search" in searchable_answer

    text = ""
    if searchable is True:
        search_query = client.text_generation(prompt=
            QUERY_PROMPT.replace("<|question|>", user_messages), stop_sequences=["\n", END_TOKEN]
        ).strip()
        search_uestion = client.text_generation(prompt=
            SEARCH_PROMPT.replace("<|question|>", user_messages), stop_sequences=["\n", END_TOKEN]
        ).strip()
        text += "```\nSearch query: " + search_query + "\n```\n\n"
        options = get_webpage_content_playwright(search_query)
        text += client.text_generation(prompt=
            USER_TOKEN + 
            options
            + "\nQuestion: "
            + search_uestion
            + "\n\nAnswer in german plain text:" + END_TOKEN + ASSISTANT_TOKEN,
            max_new_tokens=512,
            temperature=0.1,
            stop_sequences=["<|", END_TOKEN],
        )
        yield text.replace("<|", "")
    else:
        for token in client.text_generation(prompt=
            input_prompt, max_new_tokens=512, temperature=0.6, stop_sequences=["<|", END_TOKEN], stream=True
        ):
            text += token
            yield text.replace("<|", "")


def build_chat_ui():
    with gr.Blocks() as demo:
        GET_GLOBAL_HEADER()
        gr.ChatInterface(predict)

    demo.queue(concurrency_count=4)
    demo.launch(server_port=7860, **launch_args)
