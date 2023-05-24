from transformers import (
    TextIteratorStreamer,
)
from threading import Thread

from atra.model_utils.model_utils import get_model_and_processor



start_message = """
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.
"""



def convert_history_to_text(history):
    text = start_message + "".join(
        [
            "".join(
                [
                    f"<|prompter|>{item[0]}<|endoftext|>",
                    f"<|assistant|>{item[1]}<|endoftext|>",
                ]
            )
            for item in history[:-1]
        ]
    )
    text += "".join(
        [
            "".join(
                [
                    f"<|prompter|>{history[-1][0]}<|endoftext|>",
                    f"<|assistant|>{history[-1][1]}<|endoftext|>",
                ]
            )
        ]
    )
    return text


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history):
    print(f"history: {history}")
    m, tok = get_model_and_processor(lang="universal", task="chat")

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = convert_history_to_text(history)

    # Tokenize the messages string
    input_ids = tok(messages, return_tensors="pt").input_ids
    input_ids = input_ids.to(m.device)
    streamer = TextIteratorStreamer(tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=1024,
        streamer=streamer,
    )


    def generate_and_signal_complete():
        m.generate(**generate_kwargs)

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()


    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = partial_text
        yield history