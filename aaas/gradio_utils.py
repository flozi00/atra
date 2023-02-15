from datetime import timedelta
import os

import gradio as gr
from transformers.pipelines.audio_utils import ffmpeg_read

from aaas.statics import LANG_MAPPING, TO_VAD, TO_OCR
from aaas.datastore import add_to_queue, get_transkript, set_transkript
from aaas.silero_vad import silero_vad
import numpy as np

from aaas.text_utils import question_answering

langs = sorted(list(LANG_MAPPING.keys()))

model_vad, get_speech_timestamps = silero_vad(True)


def build_edit_ui():
    task_id = gr.Textbox(label="Task ID", max_lines=3)

    audio_file = gr.Audio(label="Audiofile")
    transcription = gr.Textbox(max_lines=10)
    send = gr.Button(value="Send")

    task_id.change(
        fn=get_audio,
        inputs=task_id,
        outputs=[audio_file, transcription],
        api_name="get_audio",
    )

    send.click(
        fn=set_transkript,
        inputs=[task_id, transcription],
        outputs=[],
        api_name="correct_transcription",
    )


def build_subtitle_ui():
    with gr.Row():
        video_file_in = gr.Video(source="upload", type="filepath", label="VideoFile")
        video_file_out = gr.Video(source="upload", type="filepath", label="VideoFile")

    task_id = gr.Textbox(label="Task ID", max_lines=3)

    refresh = gr.Button(value="Get Results")

    refresh.click(
        fn=get_sub_video,
        inputs=[task_id, video_file_in],
        outputs=[video_file_out],
        api_name="subtitle",
    )


def build_asr_ui():
    with gr.Row():
        lang = gr.Radio(langs, value=langs[0], label="Source Language")
        model_config = gr.Radio(
            choices=["small", "medium", "large"], value="large", label="model size"
        )

    with gr.Row():
        audio_file = gr.Audio(source="upload", type="filepath", label="Audiofile")

    task_id = gr.Textbox(label="Task ID", max_lines=3)

    audio_file.change(
        fn=add_to_vad_queue,
        inputs=[audio_file, lang, model_config],
        outputs=[task_id],
        api_name="transcription",
    )


def build_ocr_ui():
    with gr.Row():
        model_config = gr.Radio(
            choices=["small", "large"], value="large", label="model size"
        )
        ocr_mode = gr.Radio(choices=["handwritten", "printed"], label="OCR Mode")

    with gr.Row():
        image_file = gr.Image(source="upload", type="filepath", label="Imagefile")

    task_id = gr.Textbox(label="Task ID", max_lines=3)

    image_file.change(
        fn=add_to_ocr_queue,
        inputs=[image_file, model_config, ocr_mode],
        outputs=[task_id],
        api_name="ocr",
    )


def build_qa_ui():
    question = gr.Textbox(label="Question", max_lines=3)
    context = gr.Textbox(label="Context", max_lines=10)
    answer = gr.Textbox(label="Answer", max_lines=3)

    send = gr.Button(value="Send")

    send.click(
        fn=question_answering,
        inputs=[question, context],
        outputs=[answer],
        api_name="question_answering",
    )


def build_results_ui():
    task_id = gr.Textbox(label="Task ID", max_lines=3)

    with gr.Row():
        with gr.TabItem("Transcription"):
            transcription = gr.Textbox(max_lines=10)
        with gr.TabItem("details"):
            chunks = gr.JSON()

    task_id.change(
        fn=get_transcription,
        inputs=task_id,
        outputs=[transcription, chunks],
        api_name="get_transcription",
    )


def build_gradio():
    ui = gr.Blocks()

    with ui:
        with gr.Tabs():
            with gr.Tab("ASR"):
                build_asr_ui()
            with gr.Tab("OCR"):
                build_ocr_ui()
            with gr.Tab("QA"):
                build_qa_ui()
            with gr.Tab("Subtitles"):
                build_subtitle_ui()
            with gr.Tab("Results"):
                build_results_ui()
            with gr.Tab("Edit"):
                build_edit_ui()

    return ui


def add_to_ocr_queue(image, model_config, mode):
    if image is not None and len(image) > 8:

        with open(image, "rb") as f:
            payload = f.read()

        os.remove(image)

    queue = add_to_queue(
        audio_batch=[payload],
        master="",
        main_lang=mode,
        model_config=model_config,
        times=TO_OCR,
    )

    return queue[0]


def add_to_vad_queue(audio, main_lang, model_config):
    if main_lang not in langs:
        main_lang = "german"
    if model_config not in ["small", "medium", "large"]:
        model_config = "small"

    if audio is not None and len(audio) > 8:
        audio_path = audio

        with open(audio, "rb") as f:
            payload = f.read()

        audio = ffmpeg_read(payload, sampling_rate=16000)
        os.remove(audio_path)

    queue = add_to_queue(
        audio_batch=[audio.tobytes()],
        master="",
        main_lang=f"{main_lang}",
        model_config=model_config,
        times=TO_VAD,
    )

    return queue[0]


def add_vad_chunks(audio, main_lang, model_config):
    queue_string = ""
    speech_timestamps = []
    silence_duration = 500
    if main_lang not in langs:
        main_lang = "german"
    if model_config not in ["small", "medium", "large"]:
        model_config = "small"

    queue = []
    # audio = seperate_vocal(audio)

    while len(speech_timestamps) <= int((len(audio) / 16000) / 10):
        speech_timestamps = get_speech_timestamps(
            audio,
            model_vad,
            threshold=0.6,
            sampling_rate=16000,
            min_silence_duration_ms=silence_duration,
            min_speech_duration_ms=1000,
            speech_pad_ms=silence_duration * 0.1,
            return_seconds=True,
        )
        silence_duration = silence_duration * 0.9
        if silence_duration < 2:
            break
    audio_batch = [
        audio[
            int(float(speech_timestamps[st]["start"]) * 16000) : int(
                float(speech_timestamps[st]["end"]) * 16000
            )
        ].tobytes()
        for st in range(len(speech_timestamps))
    ]

    queue = add_to_queue(
        audio_batch=audio_batch,
        master=speech_timestamps,
        main_lang=f"{main_lang}",
        model_config=model_config,
    )

    queue_string = ",".join(queue)

    return queue_string


def get_transcription(queue_string: str):
    full_transcription, chunks = "", []
    queue_string = get_transkript(str(queue_string))
    if queue_string is None:
        return "", []
    elif queue_string.metas == TO_OCR:
        return queue_string.transcript, []
    else:
        queue_string = str(queue_string.transcript)
        if len(queue_string) < 5 or "***" in queue_string:
            return queue_string, []

        queue = queue_string.split(",")

        for x in range(len(queue)):
            result = get_transkript(queue[x])
            chunks.append({"id": queue[x]})
            if result is not None:
                try:
                    chunks[x]["start_timestamp"] = int(
                        float(result.metas.split(",")[0])
                    )
                    chunks[x]["stop_timestamp"] = int(float(result.metas.split(",")[1]))
                except Exception as e:
                    print(e)
                chunks[x]["text"] = result.transcript

    full_transcription = ""
    for c in chunks:
        full_transcription += c.get("text", "") + "\n"

    return full_transcription, chunks


def get_audio(task_id):
    result = get_transkript(task_id)
    return (16000, np.frombuffer(result.data, dtype=np.float32)), result.transcript


def get_sub_video(task_id, video_file):
    segments = get_transcription(task_id)[1]
    srtFilename = "subs.srt"
    if os.path.exists(srtFilename):
        os.remove(srtFilename)
    id = 0
    for segment in segments:
        startTime = (
            str(0) + str(timedelta(seconds=int(segment["start_timestamp"]))) + ",000"
        )
        endTime = (
            str(0) + str(timedelta(seconds=int(segment["stop_timestamp"]))) + ",000"
        )
        text = segment["text"]
        seg = f"{id}\n{startTime} --> {endTime}\n{text}\n\n"
        id = id + 1

        with open(srtFilename, "a+", encoding="utf-8") as srtFile:
            srtFile.write(seg)

    os.system(
        f'ffmpeg -i "{video_file}" -i watermark.png -filter_complex "[1][0]scale2ref=w=oh*mdar:h=ih*0.2[logo][video];[video][logo]overlay=main_w-overlay_w-5:5" "{video_file}watermarked.mp4"'
    )

    os.system(
        f'ffmpeg -i "{video_file}watermarked.mp4" -vf subtitles="{srtFilename}" "{video_file}.mp4"'
    )

    return f"{video_file}.mp4"
