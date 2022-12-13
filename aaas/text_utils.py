from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

from aaas.model_utils import get_model

optimum_pipes = {}


def get_optimum_pipeline(task, model_id):
    global optimum_pipes

    pipe_id = f"{task}-{model_id}"

    pipe = optimum_pipes.get(pipe_id, None)
    if pipe is not None:
        return pipe

    if task in ["translation", "summarization"]:
        model = get_model(ORTModelForSeq2SeqLM, model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

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
