from playwright.sync_api import (
    sync_playwright,
    Page,
    Browser,
    BrowserContext,
    Playwright,
)
from atra.statics import SEARCH_BACKENDS
import random
from atra.utils import timeit

@timeit
def get_first_searx_result(
    query: str, site: str = None
) -> tuple[Page, BrowserContext, Browser, Playwright]:
    search_backend = random.choice(SEARCH_BACKENDS)
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
    except Exception as e:
        search_text = ""
    try:
        results = page.locator("#urls")
        first_link = results.locator("a").all()[0]
        first_link.click()
    except Exception as e:
        print(e)
        page.close()
        context.close()
        browser.close()
        playwright.stop()

    return page, context, browser, playwright, search_text
