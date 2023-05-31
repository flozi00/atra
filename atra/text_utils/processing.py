import rapidfuzz as fuzz


def slice_into_chunks(inputs: str, size: int = 1024, overlap: int = 128):
    tokens = len(inputs)
    steps = int(tokens / size)
    stepper = 0

    chunks = []

    for i in range(steps + 1):
        start_point = stepper - (overlap if stepper > overlap + 1 else 0)
        end_point = stepper + size
        chunks.append(inputs[start_point:end_point])
        stepper += size

    return chunks


def sort_by_similarity(query: str, candidates: list, limit: int = 5):
    results = fuzz.process.extract(
        query, candidates, scorer=fuzz.fuzz.partial_token_set_ratio, limit=limit
    )
    return "        ".join([result[0] for result in results])
