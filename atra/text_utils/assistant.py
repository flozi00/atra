import json
import os
from typing import Generator, Iterable
from sentence_transformers import util
from huggingface_hub import InferenceClient
import torch
from enum import Enum
from tqdm.auto import tqdm
from atra.text_utils.prompts import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    QA_SYSTEM_PROMPT,
    SEARCH_PROMPT_PROCESSED,
    SYSTEM_PROMPT,
    TOKENS_TO_STRIP,
    USER_TOKEN,
)
from atra.text_utils.typesense_search import SemanticSearcher, Embedder
from transformers import pipeline
from atra.utilities.redis_client import cache
from atra.utilities.retrieval import get_serp, do_browsing


class Plugins(Enum):
    LOKAL = "lokal"
    SEARCH = "search"


SERP_API_KEY = os.getenv("SERP_API_KEY")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")

pipe = pipeline(
    "text-classification",
    model="flozi00/multilingual-e5-large-llm-tasks",
    device=-1,
)


@cache.cache(ttl=60 * 60 * 24 * 7)
def get_dolly_label(prompt: str) -> str:
    return pipe(prompt)[0]["label"].strip()


class Agent:
    def __init__(
        self, llm: InferenceClient, embedder: Embedder, creative: bool = False
    ) -> None:
        self.embedder = embedder
        self.llm = llm
        self.searcher = SemanticSearcher(embedder=embedder, api_key=TYPESENSE_API_KEY)
        self.temperature = 0.1 if creative is False else 0.7
        self.creative = creative

    def __call__(
        self, last_message: str, full_history: str, url: str
    ) -> Generator[str, None, None]:
        """
        Generates a response to a given message based on the given history.
        Args:
            last_message (str): The last message in the chat.
            full_history (str): The full chat history.
            url (str): The URL of the website to be searched.
        Yields:
            Generator[str, None, None]: A generator of strings representing the response.
        """
        history_no_tokens = (
            full_history.rstrip(ASSISTANT_TOKEN)
            .rstrip()
            .replace(SYSTEM_PROMPT, "")
            .strip()
        )
        yield "Classifying Plugin"
        if full_history.count(USER_TOKEN) == 1:
            search_question = last_message
        else:
            search_question = self.generate_selfstanding_query(history_no_tokens)
        plugin = self.classify_plugin(search_question)

        if (
            plugin == Plugins.SEARCH
            and (TYPESENSE_API_KEY is not None or SERP_API_KEY is not None)
            and len(search_question) > 10
        ):
            yield "Suche: " + search_question
            if TYPESENSE_API_KEY is None:
                search_query = search_question
                if len(url) > 6:
                    search_query += f" site:{url}"
                serp_passages, links, answer = get_serp(search_query, SERP_API_KEY)
                options = self.get_filtered_webpage_content(
                    search_question, links=links
                )
                serp_text = "passage: " + "\npassage: ".join(serp_passages)
                options = serp_text + options
            else:
                options = self.get_data_from_typesense(search_question)

            yield "Antworten..."
            answer = self.do_qa(search_question, options, answer)
            for text in answer:
                yield text
        else:
            answer = self.custom_generation(full_history)
            for text in answer:
                yield text

    def log_text2text(self, input: str, output: str, tasktype: str) -> None:
        """
        Logs a text2text to txt file.
        """
        input = input.replace("-->", "~~>")
        output = output.replace("-->", "~~>")

        try:
            with open(f"logging/_{tasktype}.txt", mode="r+") as file:
                content = file.read()
        except Exception:
            content = ""

        if input not in content:
            with open(f"logging/_{tasktype}.txt", mode="a+") as file:
                file.write(f"{input} --> {output}".strip())
                file.write("\n" + "*" * 20 + "\n")

    def log_preferences(self, input: str, output: str, liked: bool) -> None:
        """
        Logs input as key and output into liked or disliked.
        Stores in json file.
        """
        try:
            with open(f"logging/_dpo.json", mode="r+") as file:
                content = json.loads(file.read())
        except Exception:
            content = {}

        key = content.get(input, {"liked": [], "disliked": []})
        if output not in key["liked"] and output not in key["disliked"]:
            if liked:
                key["liked"].append(output)
            else:
                key["disliked"].append(output)

        content[input] = key

        with open(f"logging/_dpo.json", mode="w+") as file:
            file.write(json.dumps(content, indent=4))

    def classify_plugin(self, history: str) -> Plugins:
        """
        Classifies the plugin based on the given history.

        Args:
            history (str): The history to classify the plugin for.

        Returns:
            Plugins: The plugin that matches the searchable answer.
        """
        searchable_answer = get_dolly_label(history)
        self.log_text2text(input=history, output=searchable_answer, tasktype="classify")

        if searchable_answer in ["open_qa"]:
            return Plugins.SEARCH
        else:
            return Plugins.LOKAL

    def generate_selfstanding_query(self, history: str) -> str:
        """
        Generates a search question based on the given history using the LLM.

        Args:
            history (str): The history to use as context
            for generating the search question.

        Returns:
            str: The generated search question.
        """
        text = self.llm.text_generation(
            prompt=SEARCH_PROMPT_PROCESSED.replace("<|question|>", history),
            stop_sequences=["\n", END_TOKEN],
            temperature=0.1,
            do_sample=False,
        )

        for _ in TOKENS_TO_STRIP:
            for token in TOKENS_TO_STRIP:
                text = text.rstrip(token).rstrip()

        self.log_text2text(input=history, output=text, tasktype="selfquery")

        return text

    def do_qa(self, question: str, context: str, serp_answer: str) -> Iterable[str]:
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
            QA_SYSTEM_PROMPT
            + "\n"
            + USER_TOKEN
            + "\n"
            + context
            + "\n\nFrage: "
            + question
            + "\n\n"
            + END_TOKEN
            + ASSISTANT_TOKEN
            + " Antwort: "
        )

        answer = self.llm.text_generation(
            prompt=QA_Prompt,
            max_new_tokens=512,
            temperature=self.temperature,
            stop_sequences=[END_TOKEN, "###"],
            stream=True,
            do_sample=self.creative,
        )

        for token in answer:
            text += token
            yield text

        for _ in TOKENS_TO_STRIP:
            for token in TOKENS_TO_STRIP:
                text = text.rstrip(token).rstrip()

        self.log_text2text(
            input=QA_Prompt.replace(QA_SYSTEM_PROMPT + "\n", ""),
            output=text,
            tasktype="qa",
        )
        yield text.strip()

    def re_ranking(self, query: str, options: list) -> str:
        """
        Re-ranks a list of options based on their similarity to a given query.

        Args:
            query (str): The query to compare the options against.
            options (list): A list of strings representing the options to be ranked.

        Returns:
            str: A string containing the top-ranked options that have a
            cosine similarity score greater than 0.7.
        """
        corpus = ["passage: " + o for o in options]
        query = "query: " + query

        filtered_corpus = []

        corpus_embeddings = [self.embedder(corp) for corp in corpus]
        query_embedding = self.embedder(query)

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=20 if len(corpus) > 20 else len(corpus))

        for score, idx in zip(top_results[0], top_results[1]):
            if score > 0.7:
                filtered_corpus.append(corpus[idx])

        return "\n\n".join(filtered_corpus)

    def get_data_from_typesense(self, query: str) -> str:
        """
        Searches for data in Typesense using the given query and returns an
        iterable of passages containing the query.

        Args:
            query (str): The query to search for in Typesense.

        Yields:
            Iterable[str]: An iterable of passages containing the query.
        """
        options = ""
        result = self.searcher.semantic_search(query)
        for res in result:
            for key, value in res.items():
                options += "passage: " + value + "\n\n"

        return options

    def get_filtered_webpage_content(self, query: str, links: list) -> Iterable[str]:
        """
        Uses Playwright to launch a Chromium browser and navigate
        to a search engine URL with the given query.
        Returns the filtered and re-ranked text content of the webpage.

        Args:
        - query (str): The search query to be used in the URL.

        Returns:
        - filtered (str): The filtered and re-ranked text content of the webpage.
        """
        content = ""
        for link in tqdm(links[:5]):
            content += do_browsing(link) + "\n"
        content = content.split("\n")
        filtered = ""
        for co in content:
            if len(co.split(" ")) > 16:
                filtered += co + "\n"

        filtered_words = filtered.split(" ")
        filtered = []
        STEPSIZE = 250
        for i in range(0, len(filtered_words), STEPSIZE):
            filtered.append(
                " ".join(filtered_words[i : i + int(STEPSIZE * 1.3)]).strip()
            )

        filtered = self.re_ranking(query, filtered)
        filtered = "\npassage: ".join(filtered.split("passage: ")[:3])

        return filtered

    def custom_generation(self, query) -> Iterable[str]:
        text = ""
        result = self.llm.text_generation(
            prompt=query,
            max_new_tokens=512,
            temperature=self.temperature,
            stop_sequences=[END_TOKEN, "###"],
            stream=True,
            do_sample=self.creative,
        )
        for token in result:
            text += token

            yield text

        for _ in TOKENS_TO_STRIP:
            for token in TOKENS_TO_STRIP:
                text = text.rstrip(token).rstrip()

        yield text.strip()
