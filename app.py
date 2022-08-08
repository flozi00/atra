import subprocess

import gradio as gr
import numpy as np
import torch
from pyctcdecode import build_ctcdecoder
from transformers import AutoModelForCTC, Wav2Vec2Processor
import time
import onnxruntime as rt
from pathlib import Path
import glob



def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        ) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError(
            "ffmpeg was not found but is required to load audio files from filename"
        ) from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


def run_transcription(audio, main_lang, hotword_categories):
    logs = ""
    start_time = time.time()
    transcription = ""
    chunks = []
    decoder = decoders[main_lang]

    logs += f"init vars time: {time.time() - start_time}\n"
    start_time = time.time()

    if audio is not None:
        hotwords = []
        for h in hotword_categories:
            with open(f"{h}.txt", "r") as f:
                words = f.read().splitlines()
                for w in words:
                    if len(w) >= 3:
                        hotwords.append(w.strip())

        logs += f"init hotwords time: {time.time() - start_time}\n"
        start_time = time.time()

        with open(audio, "rb") as f:
            payload = f.read()

        logs += f"read audio time: {time.time() - start_time}\n"
        start_time = time.time()

        audio = ffmpeg_read(payload, sampling_rate=16000)

        logs += f"convert audio time: {time.time() - start_time}\n"
        start_time = time.time()

        speech_timestamps = get_speech_timestamps(audio, model_vad, sampling_rate=16000)
        audio_batch = [
            audio[speech_timestamps[st]["start"] : speech_timestamps[st]["end"]]
            for st in range(len(speech_timestamps))
        ]

        logs += f"get speech timestamps time: {time.time() - start_time}\n"
        start_time = time.time()

        for x in range(0, len(audio_batch), batch_size):
            data = audio_batch[x : x + batch_size]

            input_values = processor(
                data, sampling_rate=16000, return_tensors="pt", padding=True
            ).input_values

            logs += f"process audio time: {time.time() - start_time}\n"
            start_time = time.time()

            onnx_outputs = session.run(
                None, {session.get_inputs()[0].name: input_values.numpy()}
            )

            logs += f"inference time onnx: {time.time() - start_time}\n"
            start_time = time.time()

            for y in range(len(onnx_outputs[0])):
                beams = decoder.decode_beams(
                    onnx_outputs[0][y],
                    hotword_weight=2,
                    hotwords=hotwords,
                )

                offset = (
                    speech_timestamps[x + y]["start"]
                    / processor.feature_extractor.sampling_rate
                )

                top_beam = beams[0]
                transcription_beam, lm_state, indices, logit_score, lm_score = top_beam
                transcription_beam = transcription_beam.replace('"', "")
                transcription += transcription_beam + " "
                word_offsets = []
                chunk_offset = indices
                for word, (start_offset, end_offset) in chunk_offset:
                    word_offsets.append(
                        {
                            "word": word,
                            "start_offset": start_offset,
                            "end_offset": end_offset,
                        }
                    )

                for item in word_offsets:
                    start = item["start_offset"] * model.config.inputs_to_logits_ratio
                    start /= processor.feature_extractor.sampling_rate

                    stop = item["end_offset"] * model.config.inputs_to_logits_ratio
                    stop /= processor.feature_extractor.sampling_rate

                    chunks.append(
                        {
                            "text": item["word"],
                            "timestamp": (start + offset, stop + offset),
                        }
                    )

            logs += f"LM decode time: {time.time() - start_time}\n"
            start_time = time.time()

        return transcription, chunks, hotwords, logs
    else:
        return "", [], [], ""


"""
read the hotword categories from the index.txt file
"""


def get_categories(langs):
    hotword_categories = []
    
    for lang in langs:
        path = f'{lang}/*.txt'
        for file in glob.glob(path, recursive=True):
            hotword_categories.append(file.split(".")[0])


    return hotword_categories


"""
VAD download and initialization
"""
print("Downloading VAD model")
model_vad, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=True
)

(get_speech_timestamps, _, read_audio, *_) = utils


"""
Modell download and initialization
"""
print("Downloading ASR model")
model = AutoModelForCTC.from_pretrained("aware-ai/wav2vec2-xls-r-300m")
model.eval()

processor = Wav2Vec2Processor.from_pretrained("aware-ai/wav2vec2-xls-r-300m")

"""
onnx runtime initialization
"""
ONNX_PATH = "model.onnx"

path = Path(ONNX_PATH)

if path.is_file() == False:
    print("ONNX model not found, building model")
    audio_len = 160000

    x = torch.randn(1, audio_len, requires_grad=True)

    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        ONNX_PATH,  # where to save the model (can be a file or file-like object)
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

    
print("Loading ONNX model")
sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
session = rt.InferenceSession(ONNX_PATH, sess_options)


"""
decoder stuff
"""
vocab_dict = processor.tokenizer.get_vocab()
sorted_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

decoders = {}
langs = ["german"]


for l in langs:
    decoder = build_ctcdecoder(
        labels=list(sorted_dict.keys()),
        kenlm_model_path=f"asr-as-a-service-lms/2glm-{l}.arpa",
        # unigrams = list(sorted_dict.keys()),
    )
    decoders[l] = decoder

batch_size = 4


ui = gr.Blocks()

with ui:
    with gr.Tabs():
        with gr.TabItem("target language"):
            lang = gr.Radio(langs, value=langs[0])
        with gr.TabItem("hotword categories"):
            categories = gr.CheckboxGroup(choices=get_categories(langs))

    with gr.Tabs():
        with gr.TabItem("Microphone"):
            mic = gr.Audio(source="microphone", type="filepath")
        with gr.TabItem("File"):
            audio_file = gr.Audio(source="upload", type="filepath")

    with gr.Tabs():
        with gr.TabItem("Transcription"):
            transcription = gr.Textbox()
        with gr.TabItem("Subtitle"):
            chunks = gr.JSON()
        with gr.TabItem("Hotwords"):
            hotwordlist = gr.JSON()
        with gr.TabItem("Logs"):
            logs = gr.Textbox()

    mic.change(
        fn=run_transcription,
        inputs=[mic, lang, categories],
        outputs=[transcription, chunks, hotwordlist, logs],
    )
    audio_file.change(
        fn=run_transcription,
        inputs=[audio_file, lang, categories],
        outputs=[transcription, chunks, hotwordlist, logs],
    )


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0")
