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

def do_generation(input, constraints: list[list[str]] = None, max_len = 128):
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
            use_triton=False,
        )
        model.eval()
        model = torch.compile(model, mode="max-autotune")

    if constraints is not None:
        constraints = [tokenizer(x).input_ids for x in constraints]

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
        max_new_tokens=max_len,
        min_new_tokens = 1,
        do_sample=False,
        num_beams=1,
        temperature=0.01,
        no_repeat_ngram_size=3,
        use_cache = True,
    )
    if constraints is not None:
        generate_kwargs["force_words_ids"] = constraints
        generate_kwargs["num_beams"] = 3
        return tokenizer.batch_decode(model.generate(**generate_kwargs))[0]
    else:
        generate_kwargs["streamer"] = streamer

        def generate_and_signal_complete():
            model.generate(**generate_kwargs) # pad_token_id=tokenizer.eos_token_id

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        # Initialize an empty string to store the generated text
        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            yield partial_text