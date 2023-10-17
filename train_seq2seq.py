from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.utils import Tasks
import simplepeft.train.train
import datasets

simplepeft.train.train.ACCUMULATION_STEPS = 1


BATCH_SIZE = 4
BASE_MODEL = "t5-base"
PEFT_MODEL = "t5-base-llm-tasks"
TASK = Tasks.Text2Text
LR = 1e-5


def main():
    ds = datasets.load_dataset(
        "flozi00/LLM-Task-Classification", split="train", cache_dir="./downloadcache"
    )

    model, processor = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
        use_peft=True,
        use_bnb=False,
        lora_depth=128,
    )

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=ds,
        BATCH_SIZE=BATCH_SIZE,
        source_key="text",
        target_key="named_labels",
        prefix="classify: ",
        max_input_length=512,
        max_output_length=5,
    )

    # start training
    simplepeft.train.train.start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
        num_epochs=2,
    )


if __name__ == "__main__":
    main()
