from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

BITS = 4
pretrained_model_dir = "OpenAssistant/falcon-40b-sft-mix-1226"
quantized_model_dir = f"falcon-40b-openassistant-{BITS}-bits"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        x
    ) for x in ["Can you explain the quantization of models in simple words?",
                "Erkläre mir bitte die Quantisierung von Modellen.",
                "Schreibe ein kurzes Gedicht über Freundschaft.",
                "Wie ist das Wetter heute?",]
]

quantize_config = BaseQuantizeConfig(
    bits=BITS,
    group_size=128,  # it is recommended to set the value to 128
    desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad 
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, trust_remote_code=True)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples, batch_size=4)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)
tokenizer.save_pretrained(quantized_model_dir)

model.push_to_hub(quantized_model_dir, use_safetensors=True)
tokenizer.push_to_hub(quantized_model_dir)