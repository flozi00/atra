import typesense
import os
import torch
from torch import Tensor
import requests

from atra.utilities.stats import timeit


class Embedder:
    def __init__(self) -> None:
        self.headers = {
            "Content-Type": "application/json",
        }
        self.host = os.getenv("EMBEDDER_HOST", "http://127.0.0.1:8081")

    def __call__(self, text: str) -> Tensor:
        json_data = {
            "inputs": text,
        }
        response = requests.post(
            f"{self.host}/embed", headers=self.headers, json=json_data
        )
        return torch.tensor(response.json()[0])


class SemanticSearcher:
    def __init__(
        self, embedder: Embedder, api_key: str = None, collection="articles"
    ) -> None:
        self.embedder = embedder
        self.collection_name = collection
        if api_key is None:
            api_key = "xyz"
        self.client = typesense.Client(
            {
                "api_key": api_key,
                "nodes": [
                    {
                        "host": os.getenv("TYPESENSE_HOST") or "localhost",
                        "port": "8108",
                        "protocol": "http",
                    }
                ],
                "connection_timeout_seconds": 2,
            }
        )
        self.schema = {
            "name": self.collection_name,
            "fields": [
                {"name": "article", "type": "string"},
                {"name": "source", "type": "string"},
                {
                    "name": "embedding",
                    "type": "float[]",
                    "num_dim": 1024,
                },
            ],
        }

    def create_collection(self) -> None:
        self.client.collections.create(self.schema)

    @timeit
    def upsert_documents(self, articles: list, sources: list) -> None:
        for i in range(len(articles)):
            articles[i] = "passage: " + articles[i]
        embeddings = [self.embedder(article) for article in articles]
        for article, source, embeds in zip(articles, sources, embeddings):
            document = {
                "article": article,
                "source": source,
                "embedding": embeds.tolist(),
            }
            self.client.collections[self.collection_name].documents.upsert(document)

    def delete_collection(self):
        self.client.collections[self.collection_name].delete()

    @timeit
    def semantic_search(self, query: str) -> list:
        search_params = {
            "searches": [
                {
                    "collection": self.collection_name,
                    "q": "*",
                    "vector_query": "embedding:(%s, k:50)"
                    % self.embedder("query: " + query).tolist(),
                }
            ]
        }

        results = self.client.multi_search.perform(search_params, {})

        result = []
        for r in results["results"][0]["hits"]:
            if r["vector_distance"] < 0.7:
                result.append({r["document"]["source"]: r["document"]["article"]})

        return result
