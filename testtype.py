from atra.text_utils.typesense_search import SemanticSearcher
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("intfloat/multilingual-e5-large")

searcher = SemanticSearcher(embedder=embedder)

try:
    searcher.delete_collection()
except:
    pass
searcher.create_collection()
searcher.upsert_documents(["hello world"], ["test"])
searcher.upsert_documents(["hello flo"], ["test2"])
result = searcher.semantic_search("goodbye florian")

print(result)
