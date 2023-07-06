import re
from text_generation import Client
import os

from atra.statics import END_OF_TEXT_TOKEN, HUMAN_PREFIX, ASSISTANT_PREFIX

client = Client(os.environ.get("tgi", "http://127.0.0.1:8080"))


def do_generation(input, max_len=512):
    partial_text = ""
    for response in client.generate_stream(
        input,
        max_new_tokens=max_len,
        repetition_penalty=1.2,
        temperature=0.1,
        stop_sequences=[END_OF_TEXT_TOKEN, HUMAN_PREFIX, ASSISTANT_PREFIX, "<|"],
    ):
        if not response.token.special:
            partial_text += response.token.text
            partial_text = re.sub(r"<.*?\|.*?\|.*?>", "", partial_text)
            partial_text = partial_text.replace("<|", "")
            yield partial_text
