import gradio as gr
from aaas.audio_utils import LANG_MAPPING
import os
from aaas.text_utils import translate
from aaas.audio_utils import batch_audio_by_silence, get_speech_timestamps, model_vad, inference_asr, preprocess_audio
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
                    choices=["small", "medium", "large"], value="small"
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
        if isinstance(audio, str):
            audio_path = audio

            with open(audio, "rb") as f:
                payload = f.read()

            audio = ffmpeg_read(payload, sampling_rate=16000)
            audio = preprocess_audio(audio=audio)

        speech_timestamps = get_speech_timestamps(
            audio,
            model_vad,
            threshold=0.5,
            sampling_rate=16000,
            min_silence_duration_ms=500,
            speech_pad_ms=100,
        )
        audio_batch = [
            audio[speech_timestamps[st]["start"]-800 : speech_timestamps[st]["end"]+800]
            for st in range(len(speech_timestamps))
        ]

        #audio_batch = batch_audio_by_silence(audio_batch)

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
            
        os.remove(audio_path)

    yield full_transcription["target_text"], chunks
