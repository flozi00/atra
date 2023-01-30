from aaas.train.downloader import VCtube
import datasets, pandas
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name")
parser.add_argument("--url")
parser.add_argument("--lang")

args = parser.parse_args()

dataset_name = f"dataset/{args.name}"

if os.path.isdir(dataset_name) == False:
    vc = VCtube(dataset_name, args.url, args.lang,)
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
print(ds)
ds = ds.map(add_prefix)
ds = ds.filter(lambda x: os.path.isfile(x["audio"]))
ds = ds.cast_column("audio", datasets.Audio())
print(ds)

ds.push_to_hub(args.name, private=True)
