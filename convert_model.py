from simplepeft.models import get_model
from simplepeft.utils import Tasks
import fire


def convert_model(task, model):
    for t in Tasks:
        if t.value == task:
            task = t
            break
    model, processor = get_model(
        task=task, model_name=model, use_peft=False, push_to_hub=True
    )


if __name__ == "__main__":
    fire.Fire(convert_model)
