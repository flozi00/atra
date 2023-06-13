from atra.skills.base import SkillStorage
from atra.skills.internet_search import skill as wiki_skill
from atra.text_utils.generations import do_generation
from atra.statics import HUMAN_PREFIX, ASSISTANT_PREFIX, END_OF_TEXT_TOKEN

skills = None

def add_skills():
    global skills
    skills = SkillStorage()
    skills.add_skill(skill=wiki_skill)

start_message = f"""
- You are a helpful assistant chatbot called Open Assistant.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- Your knowledge of the trainings data is from the year 2021 and you have no access to the internet, so you are not able to call wikipedia or google.
- You can generate Texts, Posts, Lyrics, Tweets or plan holiday trips.{END_OF_TEXT_TOKEN}
"""

model, tokenizer = None, None


def convert_history_to_text(history):
    text = start_message + "".join(
        [
            "".join(
                [
                    HUMAN_PREFIX + f"{item[0]}" + END_OF_TEXT_TOKEN,
                    ASSISTANT_PREFIX + f"{item[1]}" + END_OF_TEXT_TOKEN,
                ]
            )
            for item in history[:-1]
        ]
    )
    text += "".join(
        [
            "".join(
                [
                    HUMAN_PREFIX + f"{history[-1][0]}" + END_OF_TEXT_TOKEN,
                    ASSISTANT_PREFIX,
                ]
            )
        ]
    )
    return text[-2048:], history[-1][0]

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, ethernet: bool = False):
    if skills is None:
        add_skills()
    text_history, newest_prompt = convert_history_to_text(history)

    if ethernet is True:
        answer = skills.answer(prompt=text_history, newest_prompt=newest_prompt)
    else:
        answer = False
    if answer is False:
        answer = do_generation(text_history, max_len=1024)

    for new_text in answer:
        history[-1][1] = new_text
        yield history