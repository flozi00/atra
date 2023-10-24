from playwright.sync_api import sync_playwright
import requests
import json
from atra.utilities.redis_client import cache


@cache.cache(ttl=60 * 60 * 24 * 7)
def get_serp(query: str, SERP_API_KEY: str) -> tuple[list, list, str | None]:
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query, "gl": "de", "hl": "de"})
    headers = {
        "X-API-KEY": SERP_API_KEY,
        "Content-Type": "application/json",
    }

    passages = []
    answer = None
    links = []
    response = requests.request("POST", url, headers=headers, data=payload).json()
    knowledge_graph = response.get("knowledgeGraph", {})
    answer_box = response.get("answerBox", {})
    organic_results = response.get("organic", [])

    if knowledge_graph != {}:
        try:
            passages.append(knowledge_graph.get("description", ""))
        except Exception:
            pass

        try:
            passages.append(str(knowledge_graph.get("attributes", "")))
        except Exception:
            pass

    try:
        answer = answer_box.get("answer", None)
    except Exception:
        pass

    for result in organic_results:
        passages.append(result["snippet"])
        links.append(result["link"])

    return passages, links, answer


@cache.cache(ttl=60 * 60 * 24 * 7)
def do_browsing(url: str) -> str:
    content = ""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(url, timeout=3000)
            content += page.locator("body").inner_text() + "\n\n"
        except Exception as e:
            print(e, url)
        browser.close()
    return content
