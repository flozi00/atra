from atra.model_utils.model_utils import get_model_and_processor
import torch


def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def generate_embedding(sentences: list) -> torch.Tensor:
    model, tokenizer = get_model_and_processor(lang="universal", task="embedding")
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    encoded_input = encoded_input.to(model.device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = average_pool(last_hidden_states=model_output.last_hidden_state, attention_mask=encoded_input['attention_mask'])


    return sentence_embeddings
