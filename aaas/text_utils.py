from optimum.pipelines import pipeline
from aaas.audio_utils import LANG_MAPPING
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import onnxruntime

optimum_pipes = {}

provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else "CPUExecutionProvider"

def get_optimum_pipeline(task, model_id):
    global optimum_pipes

    onnx_id = model_id.split("/")[1]
    pipe_id = f"{task}-{model_id}"

    pipe = optimum_pipes.get(pipe_id, None)
    if pipe != None:
        return pipe

    if task in ["translation", "summarization"]:
        try:
            model = ORTModelForSeq2SeqLM.from_pretrained(onnx_id, provider=provider)
            tokenizer = AutoTokenizer.from_pretrained(onnx_id)
        except:
            model = ORTModelForSeq2SeqLM.from_pretrained(
                model_id, from_transformers=True, provider=provider
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, from_transformers=True)
            model.save_pretrained(onnx_id)
            tokenizer.save_pretrained(onnx_id)

    pipe = pipeline(task, model=model, tokenizer=tokenizer)
    optimum_pipes[pipe_id] = pipe

    return pipe


def translate(text, source, target):

    if source == target:
        return text

    model_id = f"Helsinki-NLP/opus-mt-{source}-{target}"
    trans_pipe = get_optimum_pipeline("translation", model_id)

    translated = trans_pipe(text)[0]["translation_text"]

    return translated


def summarize(main_lang, text):
    summarizer = get_optimum_pipeline(
        "summarization", model_id="facebook/bart-large-cnn"
    )

    en_version = translate(text, LANG_MAPPING[main_lang], "en")

    summarization = summarizer(
        en_version, max_length=130, min_length=30, do_sample=False
    )[0]["summary_text"]

    summarization = translate(summarization, "en", LANG_MAPPING[main_lang])

    return summarization
