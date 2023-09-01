from atra.text_utils.typesense_search import SemanticSearcher

searcher = SemanticSearcher()

searcher.delete_collection()
searcher.create_collection()
searcher.upsert_documents(["hello world"], ["test"])
searcher.upsert_documents(["hello flo"], ["test2"])
result = searcher.semantic_search("goodbye florian")

print(result)