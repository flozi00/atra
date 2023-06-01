from atra.skills.base import BaseSkill
import wikipedia
import re


def search_wikipedia(query) -> str:
    topic = wikipedia.search(query)[0]
    summary = wikipedia.summary(topic, sentences=5)
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
        "Tell me something about Donald Trump",
        "Wer ist Elon Musk?",
        "Erzähl mir etwas über die Bundesrepublik Deutschland",
        "wer ist donald trump"
    ],
    module=search_wikipedia,
)
