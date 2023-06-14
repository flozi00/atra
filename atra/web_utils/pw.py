from playwright.sync_api import (
    sync_playwright,
    Page,
    Browser,
    BrowserContext,
    Playwright,
)
from atra.statics import SEARCH_BACKENDS
from atra.utils import timeit

BACKEND_ID = 0


@timeit
def get_search_result(
    query: str, site: str = None, id: int = 0
) -> tuple[Page, BrowserContext, Browser, Playwright]:
    global BACKEND_ID
    search_backend = SEARCH_BACKENDS[BACKEND_ID]
    if BACKEND_ID == len(SEARCH_BACKENDS) - 1:
        BACKEND_ID = 0
    else:
        BACKEND_ID += 1

    playwright = sync_playwright().start()
    browser = playwright.firefox.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"https://{search_backend}")
    search_box = page.locator("#q")
    search_box.fill(query + ("" if site is None else f" site:{site}"))
    search_box.press("Enter")
    page.wait_for_url(f"https://{search_backend}/search*")
    try:
        search_text = page.inner_text("#answers", timeout=2000)
    except Exception:
        search_text = ""
    try:
        results = page.locator("#urls")
        first_link = results.locator("a").all()[id]
        first_link.click()
        text = page.inner_text("body")
        url = page.url
    except Exception as e:
        text = ""
        url = ""
        print(e)
    page.close()
    context.close()
    browser.close()
    playwright.stop()

    return text, search_text, url
