import re
import torch
from text_generation import Client

client = Client("http://127.0.0.1:8080")



def do_generation(input, max_len=512):
        partial_text = ""
        for response in client.generate_stream(input, max_new_tokens=max_len):
            if not response.token.special:
                partial_text += response.token.text
                partial_text = re.sub(r"<.*?\|.*?\|.*?>", "", partial_text)
                yield partial_text