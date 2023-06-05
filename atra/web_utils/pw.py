from playwright.sync_api import (
    sync_playwright,
    Page,
    Browser,
    BrowserContext,
    Playwright,
)
import time


def get_first_searx_result(
    query: str, site: str = None
) -> tuple[Page, BrowserContext, Browser, Playwright]:
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://searx.be")
    time.sleep(1)
    page.locator("#q").fill(query + ("" if site is None else f" site:{site}"))
    time.sleep(1)
    page.locator(
        "#search_form > div.input-group.col-md-8.col-md-offset-2 > span.input-group-btn > button:nth-child(1)"
    ).click()
    time.sleep(1)
    page.locator("#result-1 > a").click()

    return page, context, browser, playwright
