import typesense
import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch


class Embedder:
    def __init__(self, model_path: str) -> None:
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_path, subfolder="onnx", provider="CUDAExecutionProvider"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def encode(self, sentences: list) -> torch.Tensor:
        tokenized_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**tokenized_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings


class SemanticSearcher:
    def __init__(self, embedder: Embedder, collection="articles") -> None:
        self.embedder = embedder
        self.collection_name = collection
        self.client = typesense.Client(
            {
                "api_key": os.getenv("TYPESENSE_API_KEY", "xyz"),
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

    def upsert_documents(self, articles: list, sources: list) -> None:
        for i in range(len(articles)):
            articles[i] = "passage: " + articles[i]
        embeddings = self.embedder.encode(articles)
        for article, source, embeds in zip(articles, sources, embeddings):
            document = {
                "article": article,
                "source": source,
                "embedding": embeds.tolist(),
            }
            self.client.collections[self.collection_name].documents.upsert(document)

    def delete_collection(self):
        self.client.collections[self.collection_name].delete()

    def semantic_search(self, query: str) -> list:
        search_params = {
            "searches": [
                {
                    "collection": self.collection_name,
                    "q": "*",
                    "vector_query": "embedding:(%s, k:50)"
                    % self.embedder.encode(["query: " + query])[0].tolist(),
                }
            ]
        }

        results = self.client.multi_search.perform(search_params, {})

        result = []
        for r in results["results"][0]["hits"]:
            if r["vector_distance"] < 0.7:
                result.append({r["document"]["source"]: r["document"]["article"]})

        return result
