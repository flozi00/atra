from aaas.train.download_realworld import VCtube
from aaas.train.filter_data import similarity
import datasets, pandas
import librosa

dataset_name = "dataset"
vc = VCtube(
    dataset_name,
    "https://www.youtube.com/watch?v=tfS3giG1dIg&list=PLDX3iJGAOVck863JUHOPSxxAe90oE9mFU",
    "de",
)
vc.download_audio()  # download audios from youtube
vc.download_captions()  # download captions from youtube
vc.audio_split()

pds = pandas.read_csv(
    f"./{dataset_name}/metadata.csv", sep="|", names=["audio", "text"]
)
ds = datasets.Dataset.from_pandas(pds)
print(ds)

ds = ds.filter(
    lambda x: similarity(
        x["text"],
        librosa.load(f"./{dataset_name}/wavs/" + x["audio"], sr=16000)[0],
        "german",
        "large",
    ),
    batch_size=1,
).cast_column("audio", datasets.Audio())

print(ds)

ds.save_to_disk(f"./filtered_dataset")