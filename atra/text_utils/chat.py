from atra.skills.base import SkillStorage
from atra.skills.wiki_sums import skill as wiki_skill
from atra.text_utils.generations import do_generation
from atra.statics import HUMAN_PREFIX, ASSISTANT_PREFIX, END_OF_TEXT_TOKEN

skills = SkillStorage()
skills.add_skill(skill=wiki_skill)

start_message = f"""
- You are a helpful assistant chatbot called Open Assistant.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- Your knowledge of the trainings data is from the year 2021 and you have no access to the internet, so you are not able to call wikipedia or google.
- You are a chatbot and will not be able to do anything other than chat with the user.
- Your answers are as short as possible and precise and the language is that from the user.{END_OF_TEXT_TOKEN}
"""

model, tokenizer = None, None


def convert_history_to_text(history):
    text = start_message + "".join(
        [
            f"{END_OF_TEXT_TOKEN}".join(
                [
                    f"{HUMAN_PREFIX}{item[0]}",
                    f"{ASSISTANT_PREFIX}{item[1]}",
                ]
            )
            for item in history[:-1]
        ]
    )
    text += "".join(
        [
            f"{END_OF_TEXT_TOKEN}".join(
                [
                    f"{HUMAN_PREFIX}{history[-1][0]}",
                    f"{ASSISTANT_PREFIX}{history[-1][1]}",
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