import datasets
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.utils import Tasks
from datasets import Dataset
from peft import PeftModelForCausalLM
import simplepeft.train.train
import os

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
BASE_MODEL = os.getenv("BASE_MODEL", "HuggingFaceH4/zephyr-7b-alpha")
PEFT_MODEL = os.getenv("PEFT_MODEL", "Mistral-zephyr-german-assistant-v1")
TASK = Tasks.TEXT_GEN
LR = 1e-5

SEQ_LENGTH = int(os.getenv("SEQ_LENGTH", 4096))


def combine_strings(strings) -> list:
    result = []
    current_string = strings[0]
    for string in strings[1:]:
        if len(current_string + string) <= SEQ_LENGTH * 3:
            current_string += string
        else:
            result.append(current_string)
            current_string = string
    result.append(current_string)
    return result


def main():
    ds = datasets.load_dataset(
        os.getenv("DATASET_PATH", "flozi00/conversations"), split="train"
    )

    # load model, processor by using the get_model function
    model, processor = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
        use_peft=True,
        use_flash_v2=True,
        use_bnb=True,
        lora_depth=int(os.getenv("LORA_DEPTH", 128)),
    )

    model: PeftModelForCausalLM = model

    ds_part = Dataset.from_dict(
        {
            "conversations": combine_strings(
                ds[os.getenv("DATASET_COLUMN", "conversations")]
            ),
        }
    )

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,
        processor=processor,
        datas=ds_part,
        BATCH_SIZE=BATCH_SIZE,
        max_input_length=SEQ_LENGTH,
        text_key="conversations",
    )

    # start training
    simplepeft.train.train.start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
    )


if __name__ == "__main__":
    main()
