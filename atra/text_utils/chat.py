import torch
from transformers import (
    TextIteratorStreamer,
)
from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from atra.statics import MODEL_MAPPING

start_message = """
- You are a helpful assistant chatbot called Open Assistant.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
<|endoftext|>
"""

quant_conf = BitsAndBytesConfig(load_in_4bit=True, 
                                bnb_4bit_use_double_quant=True, 
                                bnb_4bit_quant_type="fp4",
                                bnb_4bit_compute_dtype=torch.float16)

model, tokenizer = None, None


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
    global model, tokenizer
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_MAPPING["chat"]["universal"]["name"],
            cache_dir="./model_cache",
            torch_dtype=torch.float16,
            quantization_config=quant_conf,
            #offload_folder="./model_cache",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=MODEL_MAPPING["chat"]["universal"]["name"], padding_side='left'
        )
    # Construct the input message string for the model by concatenating the current 
    # system message and conversation history
    messages = convert_history_to_text(history)

    # Tokenize the messages string
    input_ids = tokenizer(messages, return_tensors="pt", max_length=2048, truncation=True)
    input_ids = input_ids.to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        **input_ids,
        max_new_tokens=512,
        streamer=streamer,
        do_sample = False,
        num_beams = 1,
        temperature = 0.0,
        top_k = 30,
        top_p = 30,
        repetition_penalty = 1.0,
        length_penalty = 1.0,
        no_repeat_ngram_size = 5,
    )

    def generate_and_signal_complete():
        model.generate(**generate_kwargs)

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = partial_text
        yield history
