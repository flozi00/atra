import gradio as gr
from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from playwright.sync_api import sync_playwright
from text_generation import Client
import urllib.parse
from atra.text_utils.prompts import ASSISTANT_TOKEN, END_TOKEN, SYSTEM_PROMPT, SEARCH_PROMPT, CLASSIFY_SEARCHABLE, USER_TOKEN
from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('intfloat/multilingual-e5-large')

client = Client("http://127.0.0.1:8080")

def re_ranking(query, options):
    corpus = ["passage: " + o for o in options]
    query = "query: " + query

    filtered_corpus = []

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=20)

    for score, idx in zip(top_results[0], top_results[1]):
        if score > 0.7:
            filtered_corpus.append(corpus[idx])

    return "\n".join(filtered_corpus)



def get_webpage_content_playwright(query):
    url = "https://searx.be/search?categories=general&language=de&q=" + urllib.parse.quote(query)
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
    
    return users

def generate_history_as_string(history, message):
    messages = SYSTEM_PROMPT + "\n\n" + "\n".join(["\n".join([USER_TOKEN+item[0]+END_TOKEN, ASSISTANT_TOKEN+item[1]+END_TOKEN])
                          for item in history])
    
    messages += USER_TOKEN + message + END_TOKEN + ASSISTANT_TOKEN 

    return messages

def predict(message, chatbot):    
    input_prompt = generate_history_as_string(chatbot, message)
    user_messages = get_user_messages(chatbot, message)
    searchable_answer = client.generate(CLASSIFY_SEARCHABLE.replace("<|question|>", user_messages), temperature=0.1, stop_sequences=["\n"], max_new_tokens=3).generated_text
    searchable = "Search" in searchable_answer

    text = ""
    if searchable is True:
        search_query = client.generate(SEARCH_PROMPT.replace("<|question|>", user_messages), stop_sequences=["\n"]).generated_text.strip()
        text += "```\nSearch query: " + search_query + "\n```\n\n"
        options = get_webpage_content_playwright(search_query)
        text += client.generate(options + "\nQuestion: " + search_query + "\n\nAnswer in german plain text:", max_new_tokens=128, temperature=0.1).generated_text
        yield text.replace("<|", "")
    else:
        for response in client.generate_stream(input_prompt, max_new_tokens=256, temperature=0.6, stop_sequences=["<|"]):
            if not response.token.special:
                text += response.token.text
                yield text.replace("<|", "")

    
def build_chat_ui():
    with gr.Blocks() as demo:
        GET_GLOBAL_HEADER()
        gr.ChatInterface(predict)  

    demo.queue()
    demo.launch(server_port=7860, **launch_args)