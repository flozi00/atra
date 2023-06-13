import re
from atra.skills.base import SkillStorage
from atra.skills.internet_search import skill as wiki_skill
from atra.text_utils.generations import do_generation
from atra.statics import HUMAN_PREFIX, ASSISTANT_PREFIX, END_OF_TEXT_TOKEN
from atra.text_utils.language_detection import classify_language
from atra.text_utils.translation import translate

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
            f"".join(
                [
                    HUMAN_PREFIX + translate(text=f"{item[0]}", dest="English") + END_OF_TEXT_TOKEN,
                    ASSISTANT_PREFIX + translate(text=f"{item[1]}", dest="English") + END_OF_TEXT_TOKEN,
                ]
            )
            for item in history[:-1]
        ]
    )
    text += "".join(
        [
            f"".join(
                [
                    HUMAN_PREFIX + translate(text=f"{history[-1][0]}", dest="English") + END_OF_TEXT_TOKEN,
                    ASSISTANT_PREFIX,
                ]
            )
        ]
    )
    return text[-2048:]

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, ethernet: bool = False):
    text_history = convert_history_to_text(history)
    newest_prompt = history[-1][0]
    src_lang = classify_language(newest_prompt)

    if ethernet:
        answer = skills.answer(prompt=text_history)
    else:
        answer = False
    if answer is False:
        answer = do_generation(text_history, max_len=1024)

    for new_text in answer:
        history[-1][1] = new_text
        yield history
    
    if "Source:" in history[-1][1]:
        results, sources = history[-1][1].split("Source:", maxsplit=1)
    else:
        results = history[-1][1]
        sources = ""
    results = re.sub(r'http\S+', '', results)
    translated = translate(text=results, src="English", dest=src_lang)
    history[-1][1] = translated + sources
    yield history