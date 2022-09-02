from transformers import AutoModelForCTC
import torch
import os
from utils import MODEL_MAPPING


def exporting(l):
    """
    onnx runtime initialization
    """
    path = f"./{l}.onnx"
    if os.path.exists(path):
        os.remove(path)

    print("Downloading ASR model")
    model = AutoModelForCTC.from_pretrained(MODEL_MAPPING[l])
    model.eval()

    print("ONNX model not found, building model")
    audio_len = 10

    x = torch.randn(1, 16000 * audio_len, requires_grad=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,  # model being run
            x,  # model input (or a tuple for multiple inputs)
            path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=13,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {0: "batch", 1: "audio_len"},  # variable length axes
                "output": {0: "batch", 1: "audio_len"},
            },
        )
