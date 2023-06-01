import torch
from transformers import (
    TextIteratorStreamer,
)
from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from atra.statics import MODEL_MAPPING

from atra.skills.base import SkillStorage
from atra.skills.wiki_sums import skill as wiki_skill


skills = SkillStorage()
skills.add_skill(skill=wiki_skill)

start_message = """
- You are a helpful assistant chatbot called Open Assistant.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- Your knowledge of the trainings data is from the year 2021 and you have no access to the internet.
- You are a chatbot and will not be able to do anything other than chat with the user.
- For tasks that you cannot do, you will tell the user that you cannot do them but help them where to find tools to do so:
    - For Speech recognition, you will tell the user to use the ASR tool in the ATRA app (Tools --> ASR).
    - For Summarization, you will tell the user to use the Summarization tool in the ATRA app (Tools --> Summarization).
    - For Translation, you will tell the user to use the Translation tool in the ATRA app (Tools --> Translator).
    - For Question Answering, you will tell the user to use the Question Answering tool in the ATRA app (Tools --> Question Answering).
<|endoftext|>
"""

quant_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype=torch.float16,
)

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
    # check if skill can answer
    answer = skills.answer(prompt=history[-1][0])
    if answer is not False:
        history[-1][1] = answer
        yield history
    else:
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=MODEL_MAPPING["chat"]["universal"]["name"],
                torch_dtype=torch.float16,
                quantization_config=quant_conf,
                # offload_folder="./model_cache",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=MODEL_MAPPING["chat"]["universal"]["name"],
                padding_side="left",
            )
        # Construct the input message string for the model by concatenating the current
        # system message and conversation history
        messages = convert_history_to_text(history=history)

        # Tokenize the messages string
        input_ids = tokenizer(
            messages, return_tensors="pt", max_length=2048, truncation=True
        )
        input_ids = input_ids.to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            **input_ids,
            max_new_tokens=512,
            streamer=streamer,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            top_k=30,
            top_p=30,
            repetition_penalty=1.0,
            length_penalty=1.0,
            no_repeat_ngram_size=5,
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

def ranker(history):
    skill_name, score = skills.choose_skill(prompt=history[-1][0])
    skill_name = skill_name.name
    if skill_name is not False and score < 0.9:
        with open("skills.csv", "a+") as f:
            f.write(f'{history[-1][0].replace(",","")},{skill_name}\n')

def missing_skill(history):
    with open("missing.csv", "a+") as f:
        f.write(f'{history[-1][0]}\n')