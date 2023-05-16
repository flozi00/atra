from datetime import timedelta
import hashlib
import os
from atra.datastore import (
    add_to_queue,
    delete_by_hash,
    get_transkript_batch,
)
from atra.audio_utils.silero_vad import get_speech_probs, silero_vad

model_vad, get_speech_timestamps = silero_vad(True)


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
    metas = ("", "", "")
    for x in range(len(results)):
        result = results[x]
        if result is None:
            return "", []
        else:
            metas = result.metas.split(",")

        if len(queue_string) < 5 or "***" in queue_string:
            return queue_string, []

        chunks.append({"id": queue[x]})
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
