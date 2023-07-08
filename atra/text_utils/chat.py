from atra.skills.internet_search import search_in_web
from atra.text_utils.generations import do_generation
from atra.statics import HUMAN_PREFIX, ASSISTANT_PREFIX, END_OF_TEXT_TOKEN

SEQ_LEN = int(2048 * 1.2)

SYSTEM_PROMPT = ""


def convert_history_to_text(history):
    text = SYSTEM_PROMPT + "".join(
        [
            "\n".join(
                [
                    HUMAN_PREFIX + f"{item[0]}" + END_OF_TEXT_TOKEN,
                    ASSISTANT_PREFIX + f"{item[1]}" + END_OF_TEXT_TOKEN,
                ]
            )
            for item in history[:-1]
        ]
    )
    text += HUMAN_PREFIX + f"{history[-1][0]}" + END_OF_TEXT_TOKEN
    return text[-SEQ_LEN:], history[-1][0]


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, ethernet: bool = False):
    text_history, newest_prompt = convert_history_to_text(history)

    if ethernet is True:
        answer = search_in_web(history=text_history, newest_prompt=newest_prompt)
        is_answer = next(answer)
        if is_answer is False:
            answer = False
    else:
        answer = False

    if answer is False:
        print("no search done")
        answer = do_generation(text_history + ASSISTANT_PREFIX, max_len=1024)

    for new_text in answer:
        history[-1][1] = new_text
        yield history
