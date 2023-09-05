from typing import Iterable
from sentence_transformers import util, SentenceTransformer
from huggingface_hub import InferenceClient
import torch
from playwright.sync_api import sync_playwright
import urllib.parse
from enum import Enum
from tqdm.auto import tqdm
from atra.text_utils.prompts import (
    ASSISTANT_TOKEN,
    CLASSIFY_SEARCHABLE,
    END_TOKEN,
    QA_SYSTEM_PROMPT,
    SEARCH_PROMPT,
    TOKENS_TO_STRIP,
    USER_TOKEN,
)
from atra.text_utils.typesense_search import SemanticSearcher


class Plugins(Enum):
    LOKAL = "lokal"
    SEARCH = "search"
    CODING = "coding"
    WRITING = "writing"
    MATH = "math"


class Agent:
    def __init__(self, llm: InferenceClient, embedder: SentenceTransformer):
        self.embedder = embedder
        self.llm = llm
        self.searcher = SemanticSearcher(embedder=embedder)

    def classify_plugin(self, history: str) -> Plugins:
        """
        Classifies the plugin based on the given history.

        Args:
            history (str): The history to classify the plugin for.

        Returns:
            Plugins: The plugin that matches the searchable answer.
        """
        searchable_answer = self.llm.text_generation(
            prompt=CLASSIFY_SEARCHABLE.replace("<|question|>", history),
            temperature=0.1,
            stop_sequences=["\n", END_TOKEN],
            max_new_tokens=3,
        ).rstrip()

        for plugin in Plugins:
            if plugin.value.lower() in searchable_answer.lower():
                return plugin

    def generate_search_question(self, history: str) -> str:
        """
        Generates a search question based on the given history using the LLM.

        Args:
            history (str): The history to use as context for generating the search question.

        Returns:
            str: The generated search question.
        """
        text = self.llm.text_generation(
            prompt=SEARCH_PROMPT.replace("<|question|>", history),
            stop_sequences=["\n", END_TOKEN],
            temperature=0.1,
        )

        for _ in TOKENS_TO_STRIP:
            for token in TOKENS_TO_STRIP:
                text = text.rstrip(token).rstrip()

        return text

    def do_qa(self, question: str, context: str) -> Iterable[str]:
        """
        Generates an answer to a question based on the given context using the LLM.

        Args:
            question (str): The question to be answered.
            context (str): The context to use for answering the question.

        Returns:
            str: The generated answer.
        """
        text = ""
        QA_Prompt = (
            QA_SYSTEM_PROMPT + 
            USER_TOKEN
            + context
            + "\n\nFrage: "
            + question
            + "\n\nAntwort:"
            + END_TOKEN
            + ASSISTANT_TOKEN
        )
        answer = self.llm.text_generation(
            prompt=QA_Prompt,
            max_new_tokens=512,
            temperature=0.1,
            stop_sequences=[END_TOKEN, "###"],
            stream=True,
        )

        for token in answer:
            text += token
            yield text
        
        for _ in TOKENS_TO_STRIP:
            for token in TOKENS_TO_STRIP:
                text = text.rstrip(token).rstrip()
        
        yield text

    def re_ranking(self, query: str, options: list) -> str:
        """
        Re-ranks a list of options based on their similarity to a given query.

        Args:
            query (str): The query to compare the options against.
            options (list): A list of strings representing the options to be ranked.

        Returns:
            str: A string containing the top-ranked options that have a cosine similarity score greater than 0.7.
        """
        corpus = ["passage: " + o for o in options]
        query = "query: " + query

        filtered_corpus = []

        corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=20 if len(corpus) > 20 else len(corpus))

        for score, idx in zip(top_results[0], top_results[1]):
            if score > 0.7:
                filtered_corpus.append(corpus[idx])

        return "\n".join(filtered_corpus)

    def get_data_from_typesense(self, query: str) -> str:
        """
        Searches for data in Typesense using the given query and returns an iterable of passages containing the query.

        Args:
            query (str): The query to search for in Typesense.

        Yields:
            Iterable[str]: An iterable of passages containing the query.
        """
        options = ""
        result = self.searcher.semantic_search(query)
        for res in result:
            for key, value in res.items():
                options += "passage: " + value + "\n"

        return options

    def get_webpage_content_playwright(self, query: str) -> Iterable[str]:
        """
        Uses Playwright to launch a Chromium browser and navigate to a search engine URL with the given query.
        Returns the filtered and re-ranked text content of the webpage.

        Args:
        - query (str): The search query to be used in the URL.

        Returns:
        - filtered (str): The filtered and re-ranked text content of the webpage.
        """
        url = "https://duckduckgo.com/?t=h_&ia=web&q=" + urllib.parse.quote(query)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            content = page.locator("body").inner_text()
            links = page.locator("body").get_by_role("link").all()
            links = [
                link.get_attribute("href")
                for link in links
                if "https://" in link.get_attribute("href")
            ][:3]
            for link in tqdm(links):
                try:
                    yield f"Reading {link}"
                    page.goto(link, timeout=5000)
                    content += page.locator("body").inner_text()
                except Exception as e:
                    print(e, link)
            browser.close()

        content = content.split("\n")
        filtered = ""
        for co in content:
            if len(co.split(" ")) > 5:
                filtered += co + "\n"

        yield "Counting passages: " + str(len(filtered.split("\n")))
        filtered = self.re_ranking(query, filtered.split("\n"))

        yield filtered

    def custom_generation(self, query) -> Iterable[str]:
        text = ""
        result = self.llm.text_generation(
            prompt=query,
            max_new_tokens=512,
            temperature=0.1,
            stop_sequences=[END_TOKEN, "###"],
            stream=True,
        )
        for token in result:
            text += token

            yield text

        for _ in TOKENS_TO_STRIP:
            for token in TOKENS_TO_STRIP:
                text = text.rstrip(token).rstrip()

        yield text.rstrip()
