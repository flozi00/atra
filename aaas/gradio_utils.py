import gradio as gr
from aaas.audio_utils import LANG_MAPPING
from aaas.backend_utils import decrease_queue
import os
from aaas.video_utils import merge_subtitles
from aaas.text_utils import translate
from aaas.remote_utils import remote_inference
from aaas.audio_utils import batch_audio_by_silence, get_speech_timestamps, model_vad
from transformers.pipelines.audio_utils import ffmpeg_read

langs = list(LANG_MAPPING.keys())


def build_gradio():
    ui = gr.Blocks()

    with ui:
        with gr.Tabs():
            with gr.TabItem("audio language"):
                lang = gr.Radio(langs, value=langs[0])
            with gr.TabItem("model configuration"):
                model_config = gr.Radio(
                    choices=["monolingual", "multilingual"], value="monolingual"
                )
            with gr.TabItem("translate to"):
                target_lang = gr.Radio(langs)

        with gr.Tabs():
            with gr.TabItem("Microphone"):
                mic = gr.Audio(source="microphone", type="filepath")
            with gr.TabItem("File"):
                audio_file = gr.Audio(source="upload", type="filepath")

        with gr.Tabs():
            with gr.TabItem("Transcription"):
                transcription = gr.Textbox()
            with gr.TabItem("details"):
                chunks = gr.JSON()
            with gr.TabItem("Subtitled Video"):
                video = gr.Video()

        mic.change(
            fn=run_transcription,
            inputs=[mic, lang, model_config, target_lang],
            outputs=[transcription, chunks, video],
            api_name="transcription",
        )
        audio_file.change(
            fn=run_transcription,
            inputs=[audio_file, lang, model_config, target_lang],
            outputs=[transcription, chunks, video],
        )

    return ui


def run_transcription(audio, main_lang, model_config, target_lang=""):
    chunks = []
    file = None
    full_transcription = {"target_text": ""}
    if target_lang == "":
        target_lang = main_lang

    if audio is not None and len(audio) > 3:
        if isinstance(audio, str):
            audio_name = audio.split(".")[-2]
            audio_path = audio
            extension = audio_path.split(".")[-1]

            if extension in ["mp4"]:
                do_stream = True
            else:
                do_stream = False

            with open(audio, "rb") as f:
                payload = f.read()

            audio = ffmpeg_read(payload, sampling_rate=16000)

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
                    main_lang=main_lang,
                    model_config=model_config,
                    data=data,
                    premium= not do_stream,
                )
            )

        for x in range(len(audio_batch)):
            response = transcription[x][0].result()
            decrease_queue(transcription[x][1])
            chunks.append(
                {
                    "native_text": response.json(),
                    "start_timestamp": (speech_timestamps[x]["start"] / 16000) - 0.1,
                    "stop_timestamp": (speech_timestamps[x]["end"] / 16000) - 0.5,
                }
            )

        chunks = sorted(chunks, key=lambda d: d["start_timestamp"])

        for c in range(len(chunks)):
            chunks[c]["target_text"] = translate(
                chunks[c]["native_text"],
                LANG_MAPPING[main_lang],
                LANG_MAPPING[target_lang],
            )
            full_transcription["target_text"] += chunks[c]["target_text"] + "\n"

        if do_stream == True:
            file = merge_subtitles(chunks, audio_path, audio_name)

        os.remove(audio_path)

    return full_transcription["target_text"], chunks, file