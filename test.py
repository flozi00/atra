from playwright.sync_api import sync_playwright
import re
playwright = sync_playwright().start()
browser = playwright.chromium.launch(headless=False)
context = browser.new_context()

def get_first_searx_result(query:str, site: str = None) -> str:
    text = ""
    page = context.new_page()
    page.goto("https://searx.be")
    page.locator("#q").fill(query + ("" if site is None else f" site:{site}"))
    page.locator("#search_form > div.input-group.col-md-8.col-md-offset-2 > span.input-group-btn > button:nth-child(1)").click()
    page.locator("#result-1 > a").click()

    text = page.inner_text("body")
    text = text.split("\n")
    text = [line for line in text if len(line) > 32 and (len(line) / line.count(" ")) < 10]
    text = "\n".join(text)

    print(text[:5000])   

    return text


get_first_searx_result("POTUS", "en.wikipedia.org")