import jiwer
import datasets
from app import run_transcription
from tqdm.auto import tqdm


for lang in [["de", "german"]]:
    dataset = datasets.load_dataset(
        "mozilla-foundation/common_voice_11_0",
        lang[0],
        split="validation",
        streaming=True,
        use_auth_token=True,
    )
    dataset = dataset.cast_column("audio", datasets.features.Audio(sampling_rate=16000))
    errors = 0
    count = 0

    for data in tqdm(dataset):
        predicted = run_transcription(data["audio"]["array"], lang[1], [])[0]
        ground = data["sentence"]

        error = jiwer.cer(
            ground,
            predicted
        )
        errors += error
        count += 1

        print((errors/count)*100)
