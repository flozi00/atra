import datasets

logs = [
    ["_selfquery.txt", "single-queries-german"],
    ["_classify.txt", "classify-llm-tasks-german"],
    ["_qa.txt", "qa-tasks-german"],
    ["_Feedback.txt", "chat-feedback-german"],
]


def convert_file(filename, ds_name):
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
        input.append(entrie[0])
        output.append(entrie[1])

    dataset = datasets.Dataset.from_dict({"input": input, "output": output})

    dataset.push_to_hub(ds_name)


for log in logs:
    convert_file(log[0], log[1])
