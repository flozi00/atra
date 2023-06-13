from atra.skills.base import BaseSkill
from atra.statics import END_OF_TEXT_TOKEN
from atra.web_utils.pw import get_search_result
from atra.text_utils.question_answering import answer_question
import re

def search_in_web(history: str, newest_prompt: str) -> str:
    context = ""
    source = ""
    query = newest_prompt
    for x in range(0,1):
        yield "Searching in Web for the query: " + query
        wiki_page, pw_context, pw_browser, pw_, search_engine_text = get_search_result(
            query, id=x
        )
        if wiki_page.url in source:
            continue
        text = wiki_page.inner_text("body") + "\n\n" + search_engine_text
        text = text.split("\n")
        text = [
            line
            for line in text
            if len(line) > 32
            and line.count(" ") > 0
            and (len(line) / (line.count(" ") if line.count(" ") > 0 else 1)) < 10
            and ("." in line or "?" in line or "!" in line)
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
        source += wiki_page.url + "\n"
        wiki_page.close()
        pw_context.close()
        pw_browser.close()
        pw_.stop()

    yield "Generating Answer ..."
    summary = answer_question(text=context, question=query, source=source)

    if summary is False:
        return False

    for sum in summary:
        yield sum

    result, sources = sum.split("Source:", maxsplit=1)
    
    yield result + "\n\n" + "Source:" + sources


skill = BaseSkill(
    name="Internet Search",
    description="This skill uses Search engines to generate short summaries about a given topic.",
    entities={
        #"query": "Formulate a search query from the last message by attending to the chat history.",
    },
    examples=[
        "Gib mir eine Zusammenfassung über Donald Trump",
        "Was ist ein Coronavirus",
        "Suche bei Wikipedia nach dem Coronavirus",
        "Wer ist Angela Merkel",
    ],
    module=search_in_web,
)
