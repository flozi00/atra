from atra.skills.base import BaseSkill
from atra.web_utils.pw import get_first_searx_result
from atra.text_utils.question_answering import answer_question

def search_wikipedia(query: str, lang: str, prompt: str) -> str:
    wiki_page, pw_context, pw_browser = get_first_searx_result(query, f"{lang}.wikipedia.org")
    text = wiki_page.inner_text("body")
    text = text.split("\n")
    text = [line for line in text if len(line) > 32 and (len(line) / line.count(" ")) < 10]
    text = "\n".join(text)

    wiki_page.close()
    pw_context.close()
    pw_browser.close()

    summary = answer_question(text=text, question=prompt, input_lang=lang)
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
