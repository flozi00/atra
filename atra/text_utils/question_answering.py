from atra.model_utils.model_utils import get_prompt
from atra.text_utils.embedding import generate_embedding
from atra.text_utils.generations import do_generation
import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def answer_question(text, question, source: str = None, progress=gr.Progress()) -> str:
    progress.__call__(0.2, "Filtering Text")
    yield "Extracting relevant information..."
    text = sort_context(text, question)
    text = get_prompt(task="question-answering").format(text=text, question=question)
    if len(text) < 256:
        text = question
        source = None
    progress.__call__(0.8, "Answering Question")
    generated_tokens = do_generation(text, max_len=512)

    for tok in generated_tokens:
        yield tok

    if source is not None:
        yield tok + "\n\n" + "Source: " + source


def sort_context(context, prompt):
    search_index = QdrantClient(":memory:")
    id_to_use = 1

    search_index.recreate_collection(
        collection_name="qa_contexts",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    context = context.split("\n")
    steps = 4
    context_slices = [
        "\n".join(context[i : i + steps]) for i in range(0, len(context), steps)
    ]
    embeddings = generate_embedding(context_slices, "passage")

    for example in range(len(context_slices)):
        search_index.upsert(
            collection_name="qa_contexts",
            points=[
                PointStruct(
                    id=id_to_use,
                    vector=embeddings[example].tolist(),
                    payload={"text": context_slices[example]},
                )
            ],
        )
        id_to_use += 1

    embeddings = generate_embedding(prompt, "query")
    search_result = search_index.search(
        collection_name="qa_contexts",
        query_vector=embeddings[0].tolist(),
        filter=None,
        top=5,
    )
    new_context = ""
    for i in range((len(search_result))):
        if len(new_context) > 2048 or search_result[i].score < 0.6:
            break
        new_context += search_result[i].payload["text"] + "\n\n"

    return new_context
