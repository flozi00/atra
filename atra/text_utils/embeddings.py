import os
import requests
from atra.utilities.redis_client import cache
from qdrant_client import QdrantClient
from qdrant_client.http import models

headers = {
    "Content-Type": "application/json",
}
host = os.getenv("EMBEDDER_HOST", "http://127.0.0.1:8081")


@cache.cache(ttl=60 * 60 * 24 * 7)
def Embedder(text: str) -> list:
    json_data = {
        "inputs": text[:1024],
    }
    response = requests.post(f"{host}/embed", headers=headers, json=json_data)
    return response.json()[0]


UUID_COUNTER = 0

client = QdrantClient(":memory:")
collection_name = "qa_collection_temp"

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
        ),
    ),
)


def reset():
    global UUID_COUNTER
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
    )
    UUID_COUNTER = 0


def add_to_collection(payload: dict):
    global UUID_COUNTER
    client.upload_records(
        collection_name=collection_name,
        records=[
            models.Record(
                id=UUID_COUNTER,
                vector=Embedder("passage: " + payload["description"]),
                payload=payload,
            )
        ],
    )
    UUID_COUNTER += 1


def search_in_collection(question: str):
    results = []
    hits = client.search(
        collection_name=collection_name,
        query_vector=Embedder(f"query: {question}"),
    )

    for hit in hits:
        results.append([hit.payload, hit.score])

    return results
