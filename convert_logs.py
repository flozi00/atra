import datasets

logs = [
    ["_selfquery.txt", "single-queries-german"],
    ["_classify.txt", "classify-llm-tasks-german"],
    ["_qa.txt", "qa-tasks-german"],
]


def convert_file(filename, ds_name):
    input, output = [], []

    with open(filename, "r") as f:
        content = f.read()

    content = content.split("********************")
    for entrie in content:
        if len(entrie) < 10:
            continue
        entrie = entrie.strip()
        entrie = entrie.split("-->")
        input.append(entrie[0])
        output.append(entrie[1])

    dataset = datasets.Dataset.from_dict({"input": input, "output": output})

    dataset.push_to_hub(ds_name)


for log in logs:
    convert_file(log[0], log[1])
