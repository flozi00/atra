import torch
from transformers import (
    TextIteratorStreamer,
)
from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from atra.statics import MODEL_MAPPING

model = None
tokenizer = None

def do_generation(input):
    global model, tokenizer
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=MODEL_MAPPING["chat"]["universal"][
                "name"
            ],
        )
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        FREE_GPU_MEM = int(torch.cuda.mem_get_info()[0] / 1024**3)-4  # in GB
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_MAPPING["chat"]["universal"]["name"],
            device_map="auto",
            low_cpu_mem_usage=True,
            use_safetensors=False,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            max_memory = {0: f"{FREE_GPU_MEM}GiB"},
        )


    # Tokenize the messages string
    input_ids = tokenizer(
        input + "<|endoftext|>", return_tensors="pt", max_length=1024, truncation=True
    )
    input_ids.pop("token_type_ids", None)
    input_ids = input_ids.to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60.0,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generate_kwargs = dict(
        **input_ids,
        max_new_tokens=1024,
        streamer=streamer,
        do_sample=False,
        num_beams=1,
        temperature=0.0,
        top_k=50,
        top_p=50,
        repetition_penalty=1.0,
        length_penalty=1.0,
        no_repeat_ngram_size=5,
    )

    def generate_and_signal_complete():
        model.generate(**generate_kwargs, pad_token_id=tokenizer.eos_token_id)

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text