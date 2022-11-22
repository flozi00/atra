from optimum.pipelines import pipeline
from aaas.statics import *
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import onnxruntime

optimum_pipes = {}

providers = [
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
]

for p in providers:
    if p in onnxruntime.get_available_providers():
        provider = p
        break

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)


def get_optimum_pipeline(task, model_id):
    global optimum_pipes

    onnx_id = model_id.split("/")[1]
    pipe_id = f"{task}-{model_id}"

    pipe = optimum_pipes.get(pipe_id, None)
    if pipe != None:
        return pipe

    if task in ["translation", "summarization"]:
        try:
            model = ORTModelForSeq2SeqLM.from_pretrained(
                onnx_id, provider=provider, session_options=sess_options
            )
            tokenizer = AutoTokenizer.from_pretrained(onnx_id)
        except:
            model = ORTModelForSeq2SeqLM.from_pretrained(
                model_id,
                from_transformers=True,
                provider=provider,
                session_options=sess_options,
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
