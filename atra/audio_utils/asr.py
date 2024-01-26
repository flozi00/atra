import torch
from text_to_num.transforms import alpha2digit

from atra.audio_utils.whisper_langs import WHISPER_LANG_MAPPING
from transformers import pipeline
import gradio as gr
import warnings
import os
from transformers import AutoModelForCausalLM


warnings.filterwarnings(action="ignore")


assistant_model = AutoModelForCausalLM.from_pretrained(
    os.getenv("SPEC_ASSISTANT_MODEL", "flozi00/distilwhisper-german-v2"),
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)

if torch.cuda.is_available():
    assistant_model = assistant_model.to("cuda:0")

assistant_model.eval()
assistant_model = torch.compile(assistant_model, mode="max-autotune")

pipe = pipeline(
    "automatic-speech-recognition",
    os.getenv("ASR_MODEL", "primeline/whisper-large-v3-german"),
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    model_kwargs={
        #"load_in_4bit": torch.cuda.is_available(),
        "attn_implementation": "sdpa",
    },
    batch_size=1,
    device=0 if torch.cuda.is_available() else -1,
)
pipe.model.eval()

pipe.model = torch.compile(pipe.model, mode="max-autotune")


def speech_recognition(data, language, progress=gr.Progress()) -> str:
    if data is None:
        return ""

    progress.__call__(progress=0.7, desc="Transcribing Audio")
    transcription = inference_asr(pipe=pipe, data=data, language=language)

    progress.__call__(progress=0.8, desc="Converting to Text")
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
        stride_length_s=(5, 0),
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
