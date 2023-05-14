import hashlib
import os
from datetime import timedelta

import gradio as gr
from transformers.pipelines.audio_utils import ffmpeg_read

from aaas.datastore import (
    add_to_queue,
    delete_by_hash,
    get_transkript_batch,
)
from aaas.silero_vad import get_speech_probs, silero_vad
from aaas.statics import LANG_MAPPING

langs = sorted(list(LANG_MAPPING.keys()))

model_vad, get_speech_timestamps = silero_vad(True)

is_admin_node = os.getenv("ADMINMODE", "false") == "true"

css = """
#hidden_stuff {display: none} 
"""


def build_asr_ui():
    """
    UI for ASR
    """
    # UI for getting audio
    with gr.Row():
        with gr.TabItem("Microphone"):
            microphone_file = gr.Audio(
                source="microphone", type="filepath", label="Audio"
            )
        with gr.TabItem("File Upload"):
            audio_file = gr.Audio(source="upload", type="filepath", label="Audiofile")

    with gr.Row():
        with gr.TabItem("Transcription"):
            transcription_finished = gr.Textbox(max_lines=10)
        with gr.TabItem("details"):
            chunks_finished = gr.JSON()

    srt_file = gr.File(label="SRT File")
    refresh = gr.Button(value="Get Subtitle File")

    # hidden UI stuff
    with gr.Row(elem_id="hidden_stuff"):
        task_id = gr.Textbox(label="Task ID", max_lines=3)
        with gr.TabItem("Transcription State"):
            with gr.Row():
                with gr.TabItem("Transcription"):
                    transcription = gr.Textbox(max_lines=10)
                with gr.TabItem("details"):
                    chunks = gr.JSON()

    audio_file.change(
        fn=add_to_vad_queue,
        inputs=[audio_file],
        outputs=[task_id],
        api_name="transcription",
    )
    microphone_file.change(
        fn=add_to_vad_queue,
        inputs=[microphone_file],
        outputs=[task_id],
    )

    task_id.change(
        fn=get_transcription,
        inputs=task_id,
        outputs=[transcription, chunks],
        api_name="get_transcription",
    )

    task_id.change(
        fn=wait_for_transcription,
        inputs=task_id,
        outputs=[transcription_finished, chunks_finished],
    )

    refresh.click(
        fn=get_subs,
        inputs=[task_id],
        outputs=[srt_file],
        api_name="subtitle",
    )


def build_translator_ui():
    with gr.Row():
        with gr.Column():
            input_lang = gr.Dropdown(langs)
            input_text = gr.Textbox(label="Input Text")

        with gr.Column():
            output_lang = gr.Dropdown(langs)
            output_text = gr.Text(label="Output Text")

    send = gr.Button(label="Translate")

    send.click(
        add_to_translation_queue,
        inputs=[input_text, input_lang, output_lang],
        outputs=[output_text],
    )


def build_gradio():
    """
    Merge all UIs into one
    """
    ui = gr.Blocks(css=css)

    with ui:
        with gr.Tabs():
            with gr.Tab("ASR"):
                build_asr_ui()
            with gr.Tab("Translator"):
                build_translator_ui()

    return ui


def add_to_translation_queue(input_text: str, input_lang: str, output_lang: str):
    input_text = input_text.encode(encoding="UTF-8")
    hs = hashlib.sha256(input_text).hexdigest()
    hs = f"{hs}.txt"

    add_to_queue(
        audio_batch=[input_text],
        hashes=[hs],
        times_list=[{"source": input_lang, "target": output_lang}],
    )

    transcript, chunks = get_transcription(hs)
    while "***" in transcript:
        transcript, chunks = get_transcription(hs)

    delete_by_hash(hs)

    return transcript


def add_to_vad_queue(audio: str):
    if audio is not None and len(audio) > 8:
        audio_path = audio

        with open(audio, "rb") as f:
            payload = f.read()

        audio = ffmpeg_read(payload, sampling_rate=16000)
        os.remove(audio_path)

        queue_string = add_vad_chunks(audio)
    else:
        queue_string = ""

    return queue_string


def add_vad_chunks(audio):
    speech_timestamps = [{"start": 0, "end": len(audio) / 16000}]
    for silence_duration in [2000, 1000, 500, 100, 10]:
        temp_times = []
        for ts in speech_timestamps:
            if (ts["end"] - ts["start"]) > 20:
                temp_audio = audio[
                    int(float(ts["start"] * 16000)) : int(float(ts["end"] * 16000))
                ]
                speech_probs = get_speech_probs(
                    temp_audio,
                    model_vad,
                    sampling_rate=16000,
                )

                speech_timestamps_iteration = get_speech_timestamps(
                    temp_audio,
                    speech_probs=speech_probs,
                    threshold=0.6,
                    sampling_rate=16000,
                    min_silence_duration_ms=silence_duration,
                    min_speech_duration_ms=500,
                    speech_pad_ms=400,
                    return_seconds=True,
                )
                for temp_ts in speech_timestamps_iteration:
                    temp_times.append(
                        {
                            "start": ts["start"] + temp_ts["start"],
                            "end": ts["start"] + temp_ts["end"],
                        }
                    )

            else:
                temp_times += [ts]

        speech_timestamps = temp_times

    audio_batch = [
        audio[
            int(float(speech_timestamps[st]["start"]) * 16000) : int(
                float(speech_timestamps[st]["end"]) * 16000
            )
        ].tobytes()
        for st in range(len(speech_timestamps))
    ]

    file_format = "wav"
    hashes = []
    for x in range(len(audio_batch)):
        # Get the audio data
        audio_data = audio_batch[x]
        hs = hashlib.sha256(f"{audio_data}".encode("utf-8")).hexdigest()
        # Add the file format to the hash
        hs = f"{hs}.{file_format}"
        # Add the hash to the list of hashes
        hashes.append(hs)

    add_to_queue(
        audio_batch=audio_batch,
        hashes=hashes,
        times_list=speech_timestamps,
    )

    queue_string = ",".join(hashes)

    return queue_string


def get_transcription(queue_string: str):
    full_transcription, chunks = "", []
    queue = queue_string.split(",")
    results = get_transkript_batch(queue_string)
    for x in range(len(results)):
        result = results[x]
        if result is None:
            return "", []
        else:
            if len(queue_string) < 5 or "***" in queue_string:
                return queue_string, []

            chunks.append({"id": queue[x]})
            if result is not None:
                metas = result.metas.split(",")
                if metas[-1] == "asr":
                    chunks[x]["start_timestamp"] = int(float(metas[0]))
                    chunks[x]["stop_timestamp"] = int(float(metas[1]))

                chunks[x]["text"] = result.transcript

    if metas[-1] == "asr":
        chunks = sorted(chunks, key=lambda d: d["start_timestamp"])

    full_transcription = ""
    for c in chunks:
        full_transcription += c.get("text", "") + "\n"

    return full_transcription, chunks


def wait_for_transcription(task_id: str):
    transcript, chunks = get_transcription(task_id)
    while "***" in transcript:
        transcript, chunks = get_transcription(task_id)

    for hs in task_id.split(","):
        delete_by_hash(hs)

    return transcript, chunks


def get_subs(task_id: str):
    segments = get_transcription(task_id)[1]
    srtFilename = hashlib.sha256(task_id.encode("utf-8")).hexdigest() + ".srt"
    if os.path.exists(srtFilename):
        os.remove(srtFilename)
    id = 1
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

    return srtFilename
