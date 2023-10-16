from atra.text_utils.typesense_search import SemanticSearcher, Embedder

embedder = Embedder("intfloat/multilingual-e5-large")
print("Embedder loaded")

searcher = SemanticSearcher(embedder=embedder)

try:
    searcher.delete_collection()
except Exception:
    pass
searcher.create_collection()
for x in range(1):
    searcher.upsert_documents(["hello world"], ["test"])
    searcher.upsert_documents(["hello flo"], ["test2"])
    searcher.upsert_documents(["goodbye florian"], ["test3"])
    searcher.upsert_documents(["goodbye world"], ["test4"])
result = searcher.semantic_search("goodbye florian")

print(result)
