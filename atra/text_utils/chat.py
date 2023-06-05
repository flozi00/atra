from atra.skills.base import SkillStorage
from atra.skills.wiki_sums import skill as wiki_skill
from atra.text_utils.generations import do_generation


skills = SkillStorage()
skills.add_skill(skill=wiki_skill)

start_message = """
- You are a helpful assistant chatbot called Open Assistant.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- Your knowledge of the trainings data is from the year 2021 and you have no access to the internet.
- You are a chatbot and will not be able to do anything other than chat with the user.
<|endoftext|>
"""

model, tokenizer = None, None


def convert_history_to_text(history):
    text = start_message + "".join(
        [
            "".join(
                [
                    f"<|prompter|>{item[0]}<|endoftext|>",
                    f"<|assistant|>{item[1]}<|endoftext|>",
                ]
            )
            for item in history[:-1]
        ]
    )
    text += "".join(
        [
            "".join(
                [
                    f"<|prompter|>{history[-1][0]}<|endoftext|>",
                    f"<|assistant|>{history[-1][1]}<|endoftext|>",
                ]
            )
        ]
    )
    return text

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, ethernet: bool = False):
    # check if skill can answer
    if ethernet:
        answer = skills.answer(prompt=history[-1][0])
    else:
        answer = False
    if answer is False:
        answer = do_generation(convert_history_to_text(history))

    # Initialize an empty string to store the generated text
    for new_text in answer:
        history[-1][1] = new_text
        yield history


def ranker(history):
    skill_name, score = skills.choose_skill(prompt=history[-1][0])
    skill_name = skill_name.name
    if skill_name is not False and score < 0.9:
        with open("skills.csv", "a+") as f:
            f.write(f'{history[-1][0].replace(",","")},{skill_name}\n')


def missing_skill(history):
    with open("missing.csv", "a+") as f:
        f.write(f"{history[-1][0]}\n")
