from aaas.utils import timeit
from transformers import pipeline

qa_pipeline = None


@timeit
def question_answering(question, context):
    global qa_pipeline
    if qa_pipeline is None:
        qa_pipeline = pipeline(
            "question-answering", model="timpal0l/mdeberta-v3-base-squad2"
        )

    return qa_pipeline(question=question, context=context)["answer"]
