import pandas as pd
import datasets


def test_add_open_assistant():
    """
    Flatten tree structure into one row per path from root to leaf
    Also turn into human_bot prompting format:
        <human>: question <bot>: answer <human>: question2 <bot>: answer2 Etc.
    Also saves a .json locally as side-effect
    returns list of dicts, containing intput, prompt_type and source
    """
    data_file = "OpenAssistant/oasst1"
    ds = datasets.load_dataset(data_file)
    df = pd.concat([ds["train"].to_pandas(), ds["validation"].to_pandas()], axis=0)
    rows = {}
    message_ids = df["message_id"].values.tolist()
    message_tree_ids = df["message_tree_id"].values.tolist()
    parent_ids = df["parent_id"].values.tolist()
    texts = df["text"].values.tolist()
    roles = df["role"].values.tolist()

    for i in range(df.shape[0]):
        # collect all trees
        message_id = message_ids[i]
        message_tree_id = message_tree_ids[i]
        parent_id = parent_ids[i]
        text = texts[i]
        role = roles[i]
        new_data = (
            ("<|prompter|> " if role == "prompter" else "<|assistant|> ")
            + text
            + "<|endoftext|>"
        )
        entry = dict(message_id=message_id, parent_id=parent_id, text=new_data)
        if message_tree_id not in rows:
            rows[message_tree_id] = [entry]
        else:
            rows[message_tree_id].append(entry)

    all_rows = []

    for node_id in rows:
        # order responses in tree, based on message/parent relationship
        conversations = []

        list_msgs = rows[node_id]
        # find start
        while len(list_msgs):
            for i, leaf in enumerate(list_msgs):
                found = False
                parent_id = leaf["parent_id"]
                if parent_id is None:
                    # conversation starter
                    conversations.append(leaf)
                    found = True
                else:
                    for conv in conversations:
                        # find all conversations to add my message to
                        if (
                            parent_id in conv["message_id"]
                            and parent_id != conv["message_id"][-len(parent_id) :]
                        ):
                            # my message doesn't follow conversation
                            continue
                        if parent_id == conv["message_id"][-len(parent_id) :]:
                            # my message follows conversation, but fork first, so another follow-on message can do same
                            conversations.append(conv.copy())
                            conv[
                                "text"
                            ] += f"""
{leaf['text']}
"""
                            conv["message_id"] += leaf["message_id"]
                            found = True
                            break
                if found:
                    # my content was used, so nuke from list
                    del list_msgs[i]
                    break

        # now reduce down to final conversations, find the longest chains of message ids
        for i, conv in enumerate(conversations):
            for j, conv2 in enumerate(conversations):
                if i == j:
                    continue
                if conv["message_id"] and conv2["message_id"]:
                    assert conv["message_id"] != conv2["message_id"]
                    # delete the shorter conversation, if one contains the other
                    if conv["message_id"] in conv2["message_id"]:
                        conv["message_id"] = None
                    if conv2["message_id"] in conv["message_id"]:
                        conv2["message_id"] = None
        conversations = [
            c
            for c in conversations
            if c["message_id"] and c["text"].count("<|assistant|>") >= 1
        ]
        all_rows.extend(c["text"] for c in conversations)
    all_rows = {"conversations": all_rows}
    print(len(all_rows["conversations"]))
    return all_rows


ds = datasets.Dataset.from_dict(test_add_open_assistant())
ds.push_to_hub("openassistant-oasst1-flattened-filtered")
