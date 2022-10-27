import gradio as gr
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
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
    full_transcription = ""
    chunks = []
    model, processor = decoders[main_lang]

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

        if "https://" in audio:
            audio = download_audio(audio)

        if isinstance(audio, str):
            with open(audio, "rb") as f:
                payload = f.read()
            os.remove(audio)

            logs += f"read audio time: {'{:.4f}'.format(time.time() - start_time)}\n"
            start_time = time.time()

            audio = ffmpeg_read(payload, sampling_rate=16000)

            logs += f"convert audio time: {'{:.4f}'.format(time.time() - start_time)}\n"
            start_time = time.time()

        speech_timestamps = get_speech_timestamps(
            audio, model_vad, sampling_rate=16000, min_silence_duration_ms=250,
        )
        audio_batch = [
            audio[speech_timestamps[st]["start"] : speech_timestamps[st]["end"]]
            for st in range(len(speech_timestamps))
        ]

        logs += (
            f"get speech timestamps time: {'{:.4f}'.format(time.time() - start_time)}\n"
        )

        for x in range(len(audio_batch)):
            data = audio_batch[x]

            input_values = processor.feature_extractor(
                data, sampling_rate=16000, return_tensors="pt", truncation=True,
            ).input_features


            predicted_ids = model.generate(input_values)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens = True)[0]
            full_transcription += transcription
            chunks.append(
                {
                    "text": transcription,
                    "timestamp": (
                        speech_timestamps[x]["start"] / 16000,
                        speech_timestamps[x]["end"] / 16000,
                    ),
                }
            )

            yield full_transcription, chunks, hotwords, logs

        yield full_transcription, chunks, hotwords, logs
    else:
        yield "", [], [], ""


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
model stuff
"""
decoders = {}
langs = list(MODEL_MAPPING.keys())


for l in langs:
    processor = WhisperProcessor.from_pretrained(MODEL_MAPPING[l])
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_MAPPING[l])
    
    decoders[l] = (model, processor)


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
    ui.queue()
    ui.launch(server_name="0.0.0.0")
