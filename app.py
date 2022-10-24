import gradio as gr
from pyctcdecode import build_ctcdecoder
import time
from transformers import AutoProcessor
import onnxruntime as rt
from export_model import exporting
import glob
from utils import ffmpeg_read, model_vad, get_speech_timestamps, MODEL_MAPPING
import yt_dlp as youtube_dl
from shutil import move
import os

class FilenameCollectorPP(youtube_dl.postprocessor.common.PostProcessor):
    def __init__(self):
        super(FilenameCollectorPP, self).__init__(None)
        self.filenames = []
        self.tags = ""

    def run(self, information):
        self.filenames.append(information["filepath"])
        self.tags = " ".join(information["tags"][:3])
        return [], information


def download_audio(url):
    options = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320",
            }
        ],
        "outtmpl": "%(title)s.%(ext)s",
    }

    ydl = youtube_dl.YoutubeDL(options)
    filename_collector = FilenameCollectorPP()
    ydl.add_post_processor(filename_collector)
    ydl.download([url])

    fname, tags = filename_collector.filenames[0], filename_collector.tags
    move(fname, tags + " " + fname)
    fname = tags + " " + fname

    return fname

def run_transcription(audio, main_lang, hotword_categories):
    logs = ""
    start_time = time.time()
    transcription = ""
    chunks = []
    session, decoder, processor = decoders[main_lang]

    logs += f"init vars time: {'{:.4f}'.format(time.time() - start_time)}\n"
    start_time = time.time()

    if audio is not None:
        hotwords = []
        for h in hotword_categories:
            with open(f"{h}.txt", "r") as f:
                words = f.read().splitlines()
                for w in words:
                    if len(w) >= 3:
                        hotwords.append(w.strip())

        logs += f"init hotwords time: {'{:.4f}'.format(time.time() - start_time)}\n"
        start_time = time.time()

        if("https://" in audio):
            audio = download_audio(audio)

        with open(audio, "rb") as f:
            payload = f.read()
        os.remove(audio)

        logs += f"read audio time: {'{:.4f}'.format(time.time() - start_time)}\n"
        start_time = time.time()

        audio = ffmpeg_read(payload, sampling_rate=16000)

        logs += f"convert audio time: {'{:.4f}'.format(time.time() - start_time)}\n"
        start_time = time.time()

        speech_timestamps = get_speech_timestamps(
            audio, model_vad, sampling_rate=16000, min_silence_duration_ms=50
        )
        audio_batch = [
            audio[speech_timestamps[st]["start"] : speech_timestamps[st]["end"]]
            for st in range(len(speech_timestamps))
        ]

        logs += (
            f"get speech timestamps time: {'{:.4f}'.format(time.time() - start_time)}\n"
        )
        start_time = time.time()

        for x in range(len(audio_batch)):
            data = audio_batch[x]

            input_values = processor(
                data, sampling_rate=16000, return_tensors="pt", padding=True
            ).input_values

            logs += f"process audio time: {'{:.4f}'.format(time.time() - start_time)}\n"
            start_time = time.time()

            onnx_outputs = session.run(
                None, {session.get_inputs()[0].name: input_values.numpy()}
            )

            logs += (
                f"inference time onnx: {'{:.4f}'.format(time.time() - start_time)}\n"
            )
            start_time = time.time()

            for y in range(len(onnx_outputs[0])):
                beams = decoder.decode_beams(
                    onnx_outputs[0][y],
                    hotword_weight=20,
                    hotwords=hotwords,
                    # beam_width=500,
                )

                top_beam = beams[0]
                transcription_beam, lm_state, indices, logit_score, lm_score = top_beam
                transcription_beam = transcription_beam.replace('"', "")
                transcription += transcription_beam + " "

                chunks.append(
                    {
                        "text": transcription_beam,
                        "timestamp": (
                            speech_timestamps[x]["start"] / 16000,
                            speech_timestamps[x]["end"] / 16000,
                        ),
                    }
                )

            logs += f"LM decode time: {'{:.4f}'.format(time.time() - start_time)}\n"
            start_time = time.time()

        return transcription, chunks, hotwords, logs
    else:
        return "", [], [], ""


"""
read the hotword categories from the index.txt file
"""


def get_categories():
    hotword_categories = []

    path = f"**/*.txt"
    for file in glob.glob(path, recursive=True):
        if "/" in file and "-" not in file:
            hotword_categories.append(file.split(".")[0])

    return hotword_categories


"""
onnx runtime initialization
"""
sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL


"""
decoder stuff
"""
decoders = {}
langs = list(MODEL_MAPPING.keys())


for l in langs:
    processor = AutoProcessor.from_pretrained(MODEL_MAPPING[l])
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_dict = {
        k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
    }
    exporting(l)
    decoder = build_ctcdecoder(
        labels=list(sorted_dict.keys()),
        kenlm_model_path=f"./asr-as-a-service-lms/lm-{l}.arpa",
        # unigrams = list(sorted_dict.keys()),
    )
    EPS = ["CUDA", "OpenVINO", "CPU"]
    onnxsession = rt.InferenceSession(
        f"./{l}.onnx",
        sess_options,
        providers=[
            f"{ep}ExecutionProvider" for ep in EPS
        ],
    )
    decoders[l] = (onnxsession, decoder, processor)


ui = gr.Blocks()

with ui:
    with gr.Tabs():
        with gr.TabItem("target language"):
            lang = gr.Radio(langs, value=langs[0])
        with gr.TabItem("hotword categories"):
            categories = gr.CheckboxGroup(choices=get_categories())

    with gr.Tabs():
        with gr.TabItem("Microphone"):
            mic = gr.Audio(source="microphone", type="filepath")
        with gr.TabItem("File"):
            audio_file = gr.Audio(source="upload", type="filepath")
        with gr.TabItem("URL"):
            video_url = gr.Textbox()

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
    video_url.change(
        fn=run_transcription,
        inputs=[video_url, lang, categories],
        outputs=[transcription, chunks, hotwordlist, logs],
    )


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0")
