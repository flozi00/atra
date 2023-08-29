import typesense


class SemanticSearcher:
    def __init__(self, client: typesense.Client) -> None:
        self.client = client
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
                        "model_config": {"model_name": "intfloat/multilingual-e5-large"},
                    },
                },
            ],
        }

    def create_collection(self):
        self.client.collections.create(self.schema)

    def upsert_documents(self, documents):
        self.client.collections["articles"].documents.import_(documents)

    def delete_collection(self):
        self.client.collections["articles"].delete()

    def semantic_search(self, query: str):
        search_params = {
            "q": query,
            "query_by": "article",
        }

        results = self.client.collections["articles"].documents.search(
            search_params
        )

        return results

