from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import torch
from atra.model_utils.model_utils import get_model_and_processor, get_prompt
from atra.utils import timeit
import gradio as gr
from atra.statics import PROMPTS
from atra.text_utils.embedding import generate_embedding


def answer_question(text, question, input_lang, progress=gr.Progress()) -> str:
    text = sort_context(context=text, prompt=question)
    progress.__call__(0.2, "Filtering Text")
    text = get_prompt(task="question-answering",lang=input_lang).format(
        text=text, question=question
    )
    model, tokenizer = get_model_and_processor(
        input_lang, "question-answering", progress=progress
    )
    progress.__call__(0.7, "Tokenizing Text")
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    progress.__call__(0.8, "Answering Question")
    generated_tokens = inference_qa(model, inputs)
    progress.__call__(0.9, "Converting to Text")
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return result

def sort_context(context, prompt):
    search_index = QdrantClient(":memory:")
    id_to_use = 1

    search_index.recreate_collection(
        collection_name="qa_contexts",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    context_slices = [context[i : i + 1576] for i in range(0, len(context), 1024)]

    for example in context_slices:
        embeddings = generate_embedding(example)
        for x in range(len(embeddings)):
            search_index.upsert(
                collection_name="qa_contexts",
                points=[
                    PointStruct(
                        id=id_to_use,
                        vector=embeddings[x].tolist(),
                        payload={"text": example[x]},
                    )
                ],
            )
            id_to_use += 1

    embeddings = generate_embedding(prompt)
    search_result = search_index.search(
        collection_name="qa_contexts",
        query_vector=embeddings[0].tolist(),
        filter=None,
        top=3,
    )
    new_context = " ".join([x.payload["text"] for x in search_result])

    return new_context


@timeit
def inference_qa(model, inputs):
    inputs.to(model.device)

    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
        )

    return generated_tokens
