from aaas.train.downloader import VCtube
import datasets, pandas
import os

dataset_name = "dataset"
export_name = "breaking_lab"

if os.path.isdir(dataset_name) == False:
    vc = VCtube(
        dataset_name,
        "https://www.youtube.com/watch?v=4uwHP5-cfWo&list=PLDX3iJGAOVclEaX_BFGaBjTjFOyhadquW",
        "de",
    )
    vc.download_audio()  # download audios from youtube
    vc.download_captions()  # download captions from youtube
    vc.audio_split()

pds = pandas.read_csv(
    f"./{dataset_name}/metadata.csv", sep="|", names=["audio", "text"]
)


def add_prefix(x):
    x["audio"] = f"./{dataset_name}/wavs/" + x["audio"]
    return x


ds = datasets.Dataset.from_pandas(pds)
ds = ds.map(add_prefix)
ds = ds.cast_column("audio", datasets.Audio())

ds.push_to_hub(export_name, private=True, max_shard_size="100MB")
