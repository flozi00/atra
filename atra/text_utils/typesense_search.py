import typesense
import os

class SemanticSearcher:
    def __init__(self) -> None:
        self.client = typesense.Client({
            'api_key': os.getenv('TYPESENSE_API_KEY', 'xyz'),
            'nodes': [{
                'host': os.getenv('TYPESENSE_HOST') or 'localhost',
                'port': '8108',
                'protocol': 'http'
            }],
            'connection_timeout_seconds': 2
        })
        self.schema = {
            "name": "articles",
            "fields": [
                {"name": "article", "type": "string"},
                {"name": "source", "type": "string"},
                {
                    "name": "embedding",
                    "type": "float[]",
                    "embed": {
                        "from": ["article"],
                        "model_config": {"model_name": "ts/multilingual-e5-base"},
                    },
                },
            ],
        }

    def create_collection(self):
        self.client.collections.create(self.schema)

    def upsert_documents(self, articles: list, sources: list):
        for article, source in zip(articles, sources):
            document = {
                    "article": article,
                    "source": source,
                }
            self.client.collections["articles"].documents.upsert(document)

    def delete_collection(self):
        self.client.collections["articles"].delete()

    def semantic_search(self, query: str):
        search_params = {
            "q": query,
            "query_by": "embedding",
        }

        results = self.client.collections["articles"].documents.search(
            search_params
        )

        result = []
        for r in results["hits"]:
            result.append({r["document"]["source"]: r["document"]["article"]})

        return result

