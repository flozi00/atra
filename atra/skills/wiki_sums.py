from atra.skills.base import BaseSkill
import wikipedia
import re


def search_wikipedia(query: str, lang: str) -> str:
    wikipedia.set_lang(prefix=lang)
    topic = wikipedia.search(query)[0]
    summary = wikipedia.summary(topic)
    summary = re.sub(pattern="[\(\[].*?[\)\]]", repl="", string=summary)

    return summary


skill = BaseSkill(
    name="Wikipedia Summaries",
    description="This skill uses Wikipedia to generate short summaries about a given topic.",
    entities={
        "query": "extract the search-query from the given prompt, answer only the keyword / topic"
    },
    examples=[
        "Erzähl mir etwas über Angela Merkel",
        "Gib mir eine Zusammenfassung über Donald Trump",
        "Was ist ein Coronavirus",
        "Suche bei Wikipedia nach dem Coronavirus",
        "Gib mir die Wikipedia-Zusammenfassung über die CDU",
        "Wer ist Angela Merkel",
    ],
    module=search_wikipedia,
)
