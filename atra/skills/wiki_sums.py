from atra.skills.base import BaseSkill
from atra.web_utils.pw import get_first_searx_result
from atra.text_utils.question_answering import answer_question
import re

def search_wikipedia(prompt: str) -> str:
    wiki_page, pw_context, pw_browser, pw_, search_engine_text = get_first_searx_result(
        prompt
    )
    text = wiki_page.inner_text("body") + "\n" + search_engine_text
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
    source = wiki_page.url
    wiki_page.close()
    pw_context.close()
    pw_browser.close()
    pw_.stop()

    summary = answer_question(text=text, question=prompt, source=source)
    
    return summary


skill = BaseSkill(
    name="Wikipedia Summaries",
    description="This skill uses Wikipedia to generate short summaries about a given topic.",
    entities={
        #"query": "extract the search-query from the given prompt, answer only the keyword / topic"
    },
    examples=[
        "Gib mir eine Zusammenfassung über Donald Trump",
        "Was ist ein Coronavirus",
        "Suche bei Wikipedia nach dem Coronavirus",
        "Wer ist Angela Merkel",
    ],
    module=search_wikipedia,
)
