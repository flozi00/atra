from aaas.train.downloader import VCtube
import datasets, pandas
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name")
parser.add_argument("--url")
parser.add_argument("--lang")
parser.add_argument("--token")
parser.add_argument("--merges")

args = parser.parse_args()


def add_prefix(x):
    x["audio"] = f"./{dataset_name}/wavs/" + x["audio"]
    return x


if args.merges:
    data_sets = []
    for d in args.merges.split(","):
        if args.token:
            data_sets.append(
                datasets.load_dataset(d, split="train", use_auth_token=args.token)
            )
        else:
            data_sets.append(datasets.load_dataset(d, split="train"))
    ds = datasets.concatenate_datasets(data_sets)

else:
    dataset_name = f"dataset/{args.name}"
    if os.path.isdir(dataset_name) == False:
        vc = VCtube(dataset_name, args.url, args.lang,)
        vc.download_audio()  # download audios from youtube
        vc.download_captions()  # download captions from youtube
        vc.audio_split()

    pds = pandas.read_csv(
        f"./{dataset_name}/metadata.csv", sep="|", names=["audio", "text"]
    )

    ds = (
        datasets.Dataset.from_pandas(pds)
        .map(add_prefix)
        .filter(lambda x: os.path.isfile(x["audio"]))
        .cast_column("audio", datasets.Audio())
    )

if args.token:
    ds.push_to_hub(args.name, token=args.token, max_shard_size="50MB", private=True)
else:
    ds.push_to_hub(args.name, max_shard_size="50MB", private=True)
