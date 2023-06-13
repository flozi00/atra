from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import datasets
from tqdm import tqdm
BITS = 3
pretrained_model_dir = "TheBloke/OpenAssistant-SFT-7-Llama-30B-HF"
quantized_model_dir = f"OpenAssistant-SFT-7-Llama-30B-{BITS}-bits-autogptq"

ds = datasets.load_dataset("flozi00/openassistant-oasst1-flattened-filtered", split="train").shuffle().select(range(100))

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        x["conversations"]
    ) for x in tqdm(ds)
]

quantize_config = BaseQuantizeConfig(
    bits=BITS,
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad 
)

try:
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, trust_remote_code=True, use_safetensors=True)
except Exception as e:
    print(e)
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, trust_remote_code=True)

    model.quantize(examples, batch_size=1, use_cuda_fp16=True, cache_examples_on_gpu=True)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)
tokenizer.save_pretrained(quantized_model_dir)

model.push_to_hub(quantized_model_dir, use_safetensors=True)
tokenizer.push_to_hub(quantized_model_dir)