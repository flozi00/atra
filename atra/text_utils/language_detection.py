from transformers import pipeline
from atra.utils import timeit
from atra.model_utils.model_utils import get_model_and_processor

@timeit
def classify_language(text) -> str:
    model, tokenizer = get_model_and_processor(
        "universal", "language-detection",
    )
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=model.device)
    return classifier(text)[0]['label']