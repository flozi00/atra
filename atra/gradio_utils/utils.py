from atra.datastore import (
    delete_by_hash,
    get_transkript_batch,
)


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
