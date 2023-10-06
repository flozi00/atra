import datasets

input, output = [], []

with open("_selfquery.txt", "r") as f:
    content = f.read()

content = content.split("********************")
for entrie in content:
    if len(entrie) < 10:
        continue
    entrie = entrie.split("</s> -->")
    input.append(entrie[0])
    output.append(entrie[1])

dataset = datasets.Dataset.from_dict({"input": input, "output": output})

dataset.push_to_hub("single-queries-german")
