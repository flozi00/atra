import torch
from transformers import (
    TextIteratorStreamer,
)
from threading import Thread

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from atra.statics import END_OF_TEXT_TOKEN, MODEL_MAPPING

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
        FREE_GPU_MEM = int(torch.cuda.mem_get_info()[0] / 1024**3)-4  # in GB
        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path=MODEL_MAPPING["chat"]["universal"]["name"],
            device_map="auto",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            max_memory = {0: f"{FREE_GPU_MEM}GiB", "cpu": "32GiB"},
            inject_fused_attention=False,
            inject_fused_mlp=False,
        )


    # Tokenize the messages string
    input_ids = tokenizer(
        input + END_OF_TEXT_TOKEN, return_tensors="pt", max_length=2048, truncation=True
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
        max_new_tokens=256,
        streamer=streamer,
        do_sample=False,
        num_beams=1,
        temperature=0.1,
        no_repeat_ngram_size=3,
        use_cache = True,
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