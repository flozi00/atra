import gradio as gr
from aaas.audio_utils import (
    ffmpeg_read,
    model_vad,
    get_speech_timestamps,
    LANG_MAPPING,
    inference_asr,
    batch_audio_by_silence,
)
from aaas.video_utils import merge_subtitles
from aaas.text_utils import translate
from aaas.remote_utils import download_audio, remote_inference
import os
from fastapi import FastAPI, Request
import uvicorn
import numpy as np

app = FastAPI()

langs = list(LANG_MAPPING.keys())


@app.post("/asr/{main_lang}/{model_config}/")
async def write(request: Request, main_lang, model_config):
    mydata = await request.body()
    audio = np.frombuffer(mydata, dtype=np.float32)

    return inference_asr(
        data_batch=[audio], main_lang=main_lang, model_config=model_config
    )[0]


def run_transcription(audio, main_lang, model_config):
    chunks = []
    file = None
    full_transcription = {"text": "", "en_text": ""}

    if audio is not None and len(audio) > 3:
        if "https://" in audio:
            audio = download_audio(audio)
            do_stream = True
        else:
            do_stream = False

        if isinstance(audio, str):
            audio_name = audio.split(".")[-2]
            audio_path = audio
            if do_stream == True:
                os.system(f'demucs -n mdx_extra --two-stems=vocals "{audio}" -o out')
                audio = "./out/mdx_extra/" + audio_name + "/vocals.wav"
            with open(audio, "rb") as f:
                payload = f.read()

            audio = ffmpeg_read(payload, sampling_rate=16000)
            if do_stream == True:
                os.remove("./out/mdx_extra/" + audio_name + "/vocals.wav")
                os.remove("./out/mdx_extra/" + audio_name + "/no_vocals.wav")

        speech_timestamps = get_speech_timestamps(
            audio,
            model_vad,
            sampling_rate=16000,
            min_silence_duration_ms=250,
            speech_pad_ms=200,
        )
        audio_batch = [
            audio[speech_timestamps[st]["start"] : speech_timestamps[st]["end"]]
            for st in range(len(speech_timestamps))
        ]

        if do_stream == False:
            audio_batch = batch_audio_by_silence(audio_batch)

        transcription = []
        for data in audio_batch:
            transcription.append(
                remote_inference(
                    main_lang=main_lang, model_config=model_config, data=data
                )
            )

        for x in range(len(audio_batch)):
            response = transcription[x].result()
            chunks.append(
                {
                    "text": response.json(),
                    "start_timestamp": (speech_timestamps[x]["start"] / 16000)-0.2,
                    "stop_timestamp": (speech_timestamps[x]["end"] / 16000)-0.5,
                }
            )

        chunks = sorted(chunks, key=lambda d: d["start_timestamp"])
        for c in chunks:
            full_transcription["text"] += c["text"] + "\n"

        if do_stream == True:
            for c in range(len(chunks)):
                chunks[c]["en_text"] = translate(
                    chunks[c]["text"], LANG_MAPPING[main_lang], "en"
                )
                full_transcription["en_text"] += chunks[c]["en_text"] + "\n"

        if audio_path.split(".")[-1] in ["mp4", "webm"]:
            file = merge_subtitles(chunks, audio_path, audio_name)

        os.remove(audio_path)

    return full_transcription["text"], chunks, file


ui = gr.Blocks()

with ui:
    with gr.Tabs():
        with gr.TabItem("target language"):
            lang = gr.Radio(langs, value=langs[0])
        with gr.TabItem("model configuration"):
            model_config = gr.Radio(
                choices=["monolingual", "multilingual"], value="monolingual"
            )

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
        with gr.TabItem("details"):
            chunks = gr.JSON()
        with gr.TabItem("Subtitled Video"):
            video = gr.Video()

    mic.change(
        fn=run_transcription,
        inputs=[mic, lang, model_config],
        outputs=[transcription, chunks, video],
        api_name="transcription",
    )
    audio_file.change(
        fn=run_transcription,
        inputs=[audio_file, lang, model_config],
        outputs=[transcription, chunks, video],
    )
    video_url.change(
        fn=run_transcription,
        inputs=[video_url, lang, model_config],
        outputs=[transcription, chunks, video],
    )

app = gr.mount_gradio_app(app, ui, path="")

if __name__ == "__main__":
    uvicorn.run("app:app", port=7860, log_level="info")
