import datasets
import json

logs = [
    ["_selfquery.txt", "single-queries-german"],
    ["_classify.txt", "classify-llm-tasks-german"],
    ["_qa.txt", "qa-tasks-german"],
]

preferences = [["_dpo.json", "atra-dpo"]]


def convert_file_t2t(filename, ds_name):
    input, output = [], []

    with open(f"logging/{filename}", "r") as f:
        content = f.read()

    content = content.split("********************")
    for entrie in content:
        if len(entrie) < 10:
            continue
        if entrie[0] in input:
            continue
        entrie = entrie.strip()
        entrie = entrie.split("-->")
        input.append(entrie[0].strip())
        output.append(entrie[1].strip())

    dataset = datasets.Dataset.from_dict({"input": input, "output": output})

    dataset.push_to_hub(ds_name)


def convert_preferences(filename, ds_name):
    input, choosen, rejected = [], [], []

    with open(f"logging/{filename}", "r") as f:
        content = f.read()
    content = json.loads(content)
    keys = list(content.keys())
    keys.sort()

    for key in keys:
        likes = content[key]["liked"]
        dislikes = content[key]["disliked"]
        if len(likes) == 0 or len(dislikes) == 0:
            continue

        for like in likes:
            for dislike in dislikes:
                input.append(key)
                choosen.append(like)
                rejected.append(dislike)

    dataset = datasets.Dataset.from_dict(
        {"input": input, "choosen": choosen, "rejected": rejected}
    )

    dataset.push_to_hub(ds_name)


for log in logs:
    convert_file_t2t(log[0], log[1])

for pref in preferences:
    convert_preferences(pref[0], pref[1])
