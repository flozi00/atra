from atra.statics import HUMAN_PREFIX, END_OF_TEXT_TOKEN
from atra.text_utils.generations import do_generation
from atra.web_utils.pw import get_search_result
from atra.text_utils.question_answering import answer_question
import re


def search_in_web(history: str, newest_prompt: str) -> str:
    context = ""
    source = ""
    query = newest_prompt
    search_query = do_generation(
        f"{history + HUMAN_PREFIX} Instruction: Generate a search query to retrieve the missing information for the task given above, answer only the query {END_OF_TEXT_TOKEN}",
        max_len=32,
    )
    query = ""
    for s in search_query:
        query = s
        query = query.split(":")[-1].replace("”", "").replace("“", "")
        if len(query.split(" ")) > 8:
            yield False
            break

    if len(query) < 5:
        yield False
    else:
        yield "Searching in Web for the query: " + query
        yield "Searching in Web for the query: " + query
    text, search_engine_text, url = get_search_result(query, id=0)

    text += "\n\n" + search_engine_text
    text = re.split(r"[.!?\r\n|\r|\n]", text)

    text = [
        line
        for line in text
        if len(line) > 32
        and line.count(" ") > 0
        and (len(line) / (line.count(" ") if line.count(" ") > 0 else 1)) < 10
        and "↑" not in line
        and "^" not in line
        and "↓" not in line
        and "→" not in line
        and "←" not in line
    ]
    text = "\n".join(text)

    # regex to remove all citations, e.g. [1], [2], [3], ...
    # remove (citation needed) and [citation needed]
    text = re.sub(r"\([^()]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    context += text + "\n"
    source += url + "\n"

    yield "Generating Answer ..."
    summary = answer_question(text=context, question=newest_prompt, source=source)

    if summary is False:
        return False

    for sum in summary:
        yield sum

    if "Source:" not in sum:
        result = sum
        sources = ""
    else:
        result, sources = sum.split("Source:", maxsplit=1)

    yield result + "\n\n" + sources
