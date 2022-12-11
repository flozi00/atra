import gradio as gr
from aaas.audio_utils import LANG_MAPPING
import os
from aaas.text_utils import translate
from aaas.audio_utils import get_model_and_processor, get_speech_timestamps, model_vad, inference_asr
from transformers.pipelines.audio_utils import ffmpeg_read

langs = sorted(list(LANG_MAPPING.keys()))


def build_gradio():
    ui = gr.Blocks()

    with ui:
        with gr.Tabs():
            with gr.TabItem("audio language"):
                lang = gr.Radio(langs, value=langs[0])
            with gr.TabItem("model configuration"):
                model_config = gr.Radio(
                    choices=["small", "medium", "large"], value="large"
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

        mic.change(
            fn=run_transcription,
            inputs=[mic, lang, model_config, target_lang],
            outputs=[transcription, chunks],
            api_name="transcription",
        )
        audio_file.change(
            fn=run_transcription,
            inputs=[audio_file, lang, model_config, target_lang],
            outputs=[transcription, chunks],
        )

    return ui


def run_transcription(audio, main_lang, model_config, target_lang=""):
    chunks = []
    full_transcription = {"target_text": ""}
    if target_lang == "":
        target_lang = main_lang

    if audio is not None and len(audio) > 3:
        audio_path = audio

        with open(audio, "rb") as f:
            payload = f.read()

        audio = ffmpeg_read(payload, sampling_rate=16000)
        os.remove(audio_path)

        if(len(audio) > 29*16000):
            speech_timestamps = get_speech_timestamps(
                audio,
                model_vad,
                threshold=0.5,
                sampling_rate=16000,
                min_silence_duration_ms=500,
                speech_pad_ms=100,
            )
            audio_batch = [
                audio[speech_timestamps[st]["start"] : speech_timestamps[st]["end"]]
                for st in range(len(speech_timestamps))
            ]
        else:
            speech_timestamps = [{"start": 100, "end": len(audio)}]
            audio_batch = [audio]

        get_model_and_processor(main_lang, model_config)
        
        for x in range(len(audio_batch)):
            audio = audio_batch[x]
            response = inference_asr(
                data_batch=[audio],
                main_lang=main_lang,
                model_config=model_config,
            )[0]
            
            chunks.append(
                {
                    "native_text": response,
                    "start_timestamp": (speech_timestamps[x]["start"] / 16000) - 0.1,
                    "stop_timestamp": (speech_timestamps[x]["end"] / 16000) - 0.5,
                    "target_text": translate(response, LANG_MAPPING[main_lang], LANG_MAPPING[target_lang]),
                }
            )
            full_transcription["target_text"] += response + "\n"
            yield full_transcription["target_text"], chunks

    yield full_transcription["target_text"], chunks