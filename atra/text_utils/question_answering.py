import torch
from atra.model_utils.model_utils import get_model_and_processor
from atra.utils import timeit


def answer_question(text, question, input_lang) -> str:
    text = f"""Question: {question}
    Answer the given Question with the following Context: {text}"""
    model, tokenizer = get_model_and_processor(input_lang, "question-answering")
    input_features = tokenizer(
        text, return_tensors="pt", max_length=1024, truncation=True
    ).input_ids
    generated_tokens = inference_qa(model, input_features)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return result


@timeit
def inference_qa(model, input_features):
    if torch.cuda.is_available():
        input_features = input_features.cuda()

    with torch.inference_mode():
        generated_tokens = model.generate(
            inputs=input_features,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
        )

    return generated_tokens
