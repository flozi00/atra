import gradio as gr
from aaas.audio_utils import (
    ffmpeg_read,
    model_vad,
    get_speech_timestamps,
    LANG_MAPPING,
    get_model_and_processor,
)
from aaas.video_utils import merge_subtitles
from aaas.text_utils import translate
from aaas.remote_utils import download_audio
import torch

langs = list(LANG_MAPPING.keys())


def run_transcription(audio, main_lang, model_config):
    chunks = []
    summarization = ""
    full_transcription = {"text": "", "en_text": ""}

    if audio is not None and len(audio) > 3:
        if "https://" in audio:
            audio = download_audio(audio)

        if model_config == "multilingual":
            model, processor = get_model_and_processor("universal")
        else:
            model, processor = get_model_and_processor(main_lang)

        if isinstance(audio, str):
            with open(audio, "rb") as f:
                payload = f.read()
            audio_path = audio

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

        do_stream = len(audio_batch) > 10

        if do_stream == False:
            new_batch = []
            tmp_audio = []
            for b in audio_batch:
                if len(tmp_audio) + len(b) < 30 * 16000:
                    tmp_audio.extend(b)
                elif len(b) > 28 * 16000:
                    new_batch.append(tmp_audio)
                    tmp_audio = []
                    new_batch.append(b)
                else:
                    new_batch.append(tmp_audio)
                    tmp_audio = []

            if tmp_audio != []:
                new_batch.append(tmp_audio)

            audio_batch = new_batch

        for x in range(len(audio_batch)):
            data = audio_batch[x]

            input_values = processor.feature_extractor(
                data,
                sampling_rate=16000,
                return_tensors="pt",
                truncation=True,
            ).input_features

            with torch.inference_mode():
                if torch.cuda.is_available():
                    input_values = input_values.to("cuda").half()
                predicted_ids = model.generate(
                    input_values,
                    max_length=int(((len(data) / 16000) * 12) / 2) + 10,
                    use_cache=True,
                    no_repeat_ngram_size=1,
                    num_beams=2,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language=LANG_MAPPING[main_lang], task="transcribe"
                    ),
                )

            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

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

            if do_stream == True:
                yield full_transcription["text"], chunks, summarization, None

        if do_stream == True:
            for c in range(len(chunks)):
                chunks[c]["en_text"] = translate(
                    chunks[c]["text"], LANG_MAPPING[main_lang], "en"
                )
                full_transcription["en_text"] += chunks[c]["en_text"] + "\n"
        else:
            summarization = ""

        yield full_transcription["text"], chunks, summarization, None

        if do_stream == True:
            file = merge_subtitles(chunks, audio_path)
            yield full_transcription["text"], chunks, summarization, file
    else:
        yield "", [], "", None


ui = gr.Blocks()

with ui:
    with gr.Tabs():
        with gr.TabItem("target language"):
            lang = gr.Radio(langs, value=langs[0])
        with gr.TabItem("model configuration"):
            model_config = gr.Radio(choices=["monolingual", "multilingual"], value=["multilingual"])

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
        with gr.TabItem("Summarization"):
            sumarization = gr.Textbox()
        with gr.TabItem("Subtitled Video"):
            video = gr.Video()

    mic.change(
        fn=run_transcription,
        inputs=[mic, lang, model_config],
        outputs=[transcription, chunks, sumarization, video],
        api_name="transcription",
    )
    audio_file.change(
        fn=run_transcription,
        inputs=[audio_file, lang, model_config],
        outputs=[transcription, chunks, sumarization, video],
    )
    video_url.change(
        fn=run_transcription,
        inputs=[video_url, lang, model_config],
        outputs=[transcription, chunks, sumarization, video],
    )


if __name__ == "__main__":
    ui.queue(concurrency_count=3)
    ui.launch(server_name="0.0.0.0")
