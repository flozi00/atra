import torch
from text_to_num.transforms import alpha2digit

from atra.audio_utils.whisper_langs import WHISPER_LANG_MAPPING
from transformers import pipeline
import gradio as gr
import warnings
import os
from transformers import AutoModelForCausalLM


warnings.filterwarnings(action="ignore")

if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0).split(" ")[-1]
else:
    GPU_NAME = "CPU"

assistant_model = AutoModelForCausalLM.from_pretrained(
    os.getenv("SPEC_ASSISTANT_MODEL", "sanchit-gandhi/distil-whisper-large-v3-de-kd"),
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)

if torch.cuda.is_available():
    assistant_model = assistant_model.to("cuda:0")

# try:
#    assistant_model = torch.compile(assistant_model, mode="max-autotune")
# except Exception:
#    pass

pipe = pipeline(
    "automatic-speech-recognition",
    os.getenv("ASR_MODEL", "primeline/whisper-large-v3-german"),
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    model_kwargs={
        "load_in_4bit": True,
        # "use_flash_attention_2": "A" in GPU_NAME or "H" in GPU_NAME or "L" in GPU_NAME,
        "attn_implementation": "sdpa",
    },
    batch_size=4,
)
pipe.model.eval()

# try:
#    pipe.model = torch.compile(pipe.model, mode="max-autotune")
# except Exception:
#    pass


def speech_recognition(data, language, progress=gr.Progress()) -> str:
    if data is None:
        return ""

    progress.__call__(progress=0.8, desc="Transcribing Audio")
    transcription = inference_asr(pipe=pipe, data=data, language=language)

    progress.__call__(progress=0.9, desc="Converting to Text")
    try:
        transcription = alpha2digit(
            text=transcription, lang=WHISPER_LANG_MAPPING[language]
        )
    except Exception:
        pass

    return transcription


def inference_asr(pipe, data, language) -> str:
    generated_ids = pipe(
        data,
        chunk_length_s=30,
        stride_length_s=(10, 0),
        # return_timestamps="word",
        generate_kwargs={
            "task": "transcribe",
            "language": f"<|{WHISPER_LANG_MAPPING[language]}|>",
            "do_sample": False,
            "num_beams": 1,
            "assistant_model": assistant_model,
        },
    )
    return generated_ids["text"]
