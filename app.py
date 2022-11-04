import gradio as gr
import time
import glob
from aaas.audio_utils import (
    ffmpeg_read,
    model_vad,
    get_speech_timestamps,
    LANG_MAPPING,
    get_model_and_processor,
)
from aaas.text_utils import summarize, translate
from aaas.remote_utils import download_audio
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
langs = list(LANG_MAPPING.keys())

def run_transcription(audio, main_lang, hotword_categories):
    global trans_pipes
    logs = ""
    start_time = time.time()
    chunks = []
    summarization = ""

    model, processor = get_model_and_processor(
        main_lang, device=device
    )

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
        start_time = time.time()

        for x in range(len(audio_batch)):
            data = audio_batch[x]

            input_values = processor.feature_extractor(
                data,
                sampling_rate=16000,
                return_tensors="pt",
                truncation=True,
            ).input_features

            input_values = input_values.to(device)

            logs += (
                f"feature extractor: {'{:.4f}'.format(time.time() - start_time)}\n"
            )
            start_time = time.time()

            with torch.inference_mode():
                predicted_ids = model.generate(
                    input_values,
                    max_length=int((len(data) / 16000) * 12),
                    min_length = 4,
                    use_cache=True,
                    num_beams = 20,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language=LANG_MAPPING[main_lang], task="transcribe"
                    ),
                    return_dict_in_generate=False, output_scores=False
                )

            logs += (
                f"inference: {'{:.4f}'.format(time.time() - start_time)}\n"
            )
            start_time = time.time()

            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            logs += (
                f"decode: {'{:.4f}'.format(time.time() - start_time)}\n"
            )
            start_time = time.time()

            chunks.append(
                {
                    "text": transcription,
                    "timestamp": (
                        speech_timestamps[x]["start"] / 16000,
                        speech_timestamps[x]["end"] / 16000,
                    ),
                }
            )

            full_transcription = {"text": "", "en_text": ""}

            for c in chunks:
                full_transcription["text"] += c["text"] + "\n"

            yield full_transcription["text"], chunks, hotwords, logs, summarization

        if len(full_transcription) > 512:
            for c in range(len(chunks)):
                chunks[c]["en_text"] = translate(chunks[c]["text"], LANG_MAPPING[main_lang], "en"),
                full_transcription["en_text"] += chunks[c]["en_text"] + "\n"
            logs += (
                f"translate: {'{:.4f}'.format(time.time() - start_time)}\n"
            )
            start_time = time.time()
            summarization = summarize(full_transcription["en_text"])
        else:
            summarization = ""

        logs += f"summarization: {'{:.4f}'.format(time.time() - start_time)}\n"

        yield full_transcription["text"], chunks, hotwords, logs, summarization
    else:
        yield "", [], [], "", ""


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
            logs = gr.Textbox(lines = 10)
        with gr.TabItem("Summarization"):
            sumarization = gr.Textbox()

    mic.change(
        fn=run_transcription,
        inputs=[mic, lang, categories],
        outputs=[transcription, chunks, hotwordlist, logs, sumarization],
        api_name="transcription"
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
    ui.queue()
    ui.launch(server_name="0.0.0.0")
