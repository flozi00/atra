import gradio as gr
from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from playwright.sync_api import sync_playwright
from text_generation import Client
import urllib.parse
import json

client = Client("http://127.0.0.1:8080")

system_message = "Im Folgenden finden Sie eine Reihe von Dialogen zwischen verschiedenen Personen und einem deutschen KI-Assistenten namens Egino. Die KI versucht, hilfsbereit, höflich, ehrlich, kultiviert, gefühlsbetont und bescheiden, aber kenntnisreich zu sein. Der Assistent ist gerne bereit, bei fast allem zu helfen, und tut sein Bestes, um genau zu verstehen, was benötigt wird. Er bemüht sich auch, keine falschen oder irreführenden Informationen zu geben, und er macht Vorbehalte, wenn er sich der richtigen Antwort nicht ganz sicher ist. Dennoch ist der Assistent praktisch und tut wirklich sein Bestes, ohne sich von der Vorsicht zu sehr einschränken zu lassen. Die Antworten werden vollständig formuliert und so detailliert wie möglich sein."

SEARCH_PROMPT = """Formuliere Suchanfragen anhand der vorrausgehenden Konvesation:
Was ist Chatgpt ? --> Was ist Chatgpt
Ich habe Hunger --> Was sind schnelle Rezepte
Wer ist der aktuelle Bundespräsident --> Wer ist der aktuelle Bundespräsident
Sichere Programmierung --> Wie programmiere ich sicher
Ich suche einen guten Artikel über .net Autorisierung --> .net Autorisierung
Wer ist Jeff Bezos --> Wer ist jeff Bezos
Ich suche einen Artikel über Wallbox --> Wallbox
Wann iMac 2023 --> Wann ist das iMac 2023 Releasedatum
Überwachungskamera --> Was ist eine gute Überwachungskamera
wann kommt gta 6 raus --> Wann ist der GTA 6 Release
wer ist angela merkel ?, und wie alt ist sie --> wie alt ist angela merkel
wer ist donald trump , muss er ins gefängnis ? --> muss donald trump ins gefängnis
<|question|> -->"""

CLASSIFY_SEARCHABLE = """Klassifiziere ob die Frage im Internet gesucht werden kann oder lokal beantwortet wird:
Wer bist du ? --> Lokal
Ich habe Hunger --> Lokal
Wann iMac 2023 --> Search
Was kannst du --> Lokal
Was ist der Sinn des Lebens --> Search
Wer bist du --> Lokal
Ich suche einen Artikel über Wallbox --> Search
Wer ist Angela Merkel --> Search
Und auf Deutsch ? --> Lokal
Plane einen 3tägigen Trip nach Hawaii --> Lokal
<|question|> -->"""

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

    return filtered

def get_user_messages(history):
    users = ""
    for h in history:
        users += "," + h[0]
    
    return users

def predict(message, chatbot):    
    input_prompt = f"<|prompter|>{system_message}<|endoftext|>"
    for interaction in chatbot:
        input_prompt = "<|prompter|>" + input_prompt + str(interaction[0]) + "<|assistant|>" + str(interaction[1]) + "<|endoftext|>"

    input_prompt = input_prompt + "<|prompter|>" + str(message) + "<|endoftext|><|assistant|>"

    searchable_answer = client.generate(CLASSIFY_SEARCHABLE.replace("<|question|>", message), temperature=0.1).generated_text
    searchable = "Search" in searchable_answer

    text = ""
    if searchable is True:
        search_query = client.generate(SEARCH_PROMPT.replace("<|question|>", message), stop_sequences=["\n"]).generated_text + "\n"
        text += "Search query: " + search_query
        options = get_webpage_content_playwright(search_query)
        text += client.generate(options + "\nQuestion: " + search_query + "\n\nAnswer in german plain text:", max_new_tokens=128, temperature=0.1).generated_text
        yield text
    else:
        for response in client.generate_stream(input_prompt, max_new_tokens=256, temperature=0.6, stop_sequences=["endoftext","<|"]):
            if not response.token.special:
                text += response.token.text
                yield text

    
def build_chat_ui():
    with gr.Blocks() as demo:
        GET_GLOBAL_HEADER()
        gr.ChatInterface(predict)  

    demo.queue()
    demo.launch(server_port=7860, **launch_args)