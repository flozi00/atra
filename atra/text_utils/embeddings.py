import os
import requests
from atra.utilities.redis_client import cache

headers = {
    "Content-Type": "application/json",
}
host = os.getenv("EMBEDDER_HOST", "http://127.0.0.1:8081")


@cache.cache(ttl=60 * 60 * 24 * 7)
def Embedder(text: str) -> list:
    json_data = {
        "inputs": text,
    }
    response = requests.post(f"{host}/embed", headers=headers, json=json_data)
    return response.json()[0]
