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
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"https://{search_backend}")
    page.locator("#q").fill(query + ("" if site is None else f" site:{site}"))
    page.locator(
        "#search_form > div.input-group.col-md-8.col-md-offset-2 > span.input-group-btn > button:nth-child(1)"
    ).click()
    page.locator("#result-1 > a").click()

    return page, context, browser, playwright
