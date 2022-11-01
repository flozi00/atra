import gradio as gr
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import glob
from utils import (
    ffmpeg_read,
    model_vad,
    get_speech_timestamps,
    MODEL_MAPPING,
    LANG_MAPPING,
)
import yt_dlp as youtube_dl
from shutil import move
import os
import torch
from optimum.pipelines import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

trans_pipes = {
    "Helsinki-NLP/opus-mt-en-de": pipeline(
        "translation", model=f"Helsinki-NLP/opus-mt-en-de"
    ),
    "Helsinki-NLP/opus-mt-de-en": pipeline(
        "translation", model=f"Helsinki-NLP/opus-mt-de-en"
    ),
}


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
    global trans_pipes
    logs = ""
    start_time = time.time()
    full_transcription = ""
    chunks = []

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

        if len(hotwords) <= 1:
            hotwords = [" "]

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
            audio,
            model_vad,
            sampling_rate=16000,
            min_silence_duration_ms=250,
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
                data,
                sampling_rate=16000,
                return_tensors="pt",
                truncation=True,
            ).input_features

            input_values = input_values.to(device)

            force_words_ids = [
                processor.tokenizer(
                    hotwords, add_prefix_space=True, add_special_tokens=False
                ).input_ids
            ]

            with torch.inference_mode():
                predicted_ids = model.generate(
                    input_values,
                    # num_return_sequences=10,
                    # num_beams=10,
                    # no_repeat_ngram_size=1,
                    max_length=(len(data) / 16000) * 12,
                    use_cache=True,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language=LANG_MAPPING[main_lang], task="transcribe"
                    ),
                )

            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            full_transcription += str(transcription)
            chunks.append(
                {
                    "text": transcription,
                    "timestamp": (
                        speech_timestamps[x]["start"] / 16000,
                        speech_timestamps[x]["end"] / 16000,
                    ),
                }
            )

        if LANG_MAPPING[main_lang] != "en":
            trans_pipe = trans_pipes.get(
                f"Helsinki-NLP/opus-mt-{LANG_MAPPING[main_lang]}-en", None
            )
            if trans_pipe is None:
                trans_pipe = pipeline(
                    "translation",
                    model=f"Helsinki-NLP/opus-mt-{LANG_MAPPING[main_lang]}-en",
                )
                trans_pipes[
                    f"Helsinki-NLP/opus-mt-{LANG_MAPPING[main_lang]}-en"
                ] = trans_pipe
            en_version = trans_pipe(full_transcription)[0]["translation_text"]
        else:
            en_version = full_transcription
        summarization = summarizer(
            en_version, max_length=130, min_length=30, do_sample=False
        )[0]["summary_text"]
        if LANG_MAPPING[main_lang] != "en":
            trans_pipe = trans_pipes.get(
                f"Helsinki-NLP/opus-mt-en-{LANG_MAPPING[main_lang]}", None
            )
            if trans_pipe is None:
                trans_pipe = pipeline(
                    "translation",
                    model=f"Helsinki-NLP/opus-mt-en-{LANG_MAPPING[main_lang]}",
                )
                trans_pipes[
                    f"Helsinki-NLP/opus-mt-en-{LANG_MAPPING[main_lang]}"
                ] = trans_pipe
            summarization = trans_pipe(summarization)[0]["translation_text"]
        return full_transcription, chunks, hotwords, logs, summarization
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
model stuff
"""
decoders = {}
langs = list(MODEL_MAPPING.keys())


processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").eval()
model = model.to(device)


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
        with gr.TabItem("Summarization"):
            sumarization = gr.Textbox()

    mic.change(
        fn=run_transcription,
        inputs=[mic, lang, categories],
        outputs=[transcription, chunks, hotwordlist, logs, sumarization],
    )
    audio_file.change(
        fn=run_transcription,
        inputs=[audio_file, lang, categories],
        outputs=[transcription, chunks, hotwordlist, logs, sumarization],
    )
    video_url.change(
        fn=run_transcription,
        inputs=[video_url, lang, categories],
        outputs=[transcription, chunks, hotwordlist, logs, sumarization],
    )


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0")
