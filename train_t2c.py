import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from simplepeft.data import get_dataloader
from simplepeft.utils import Tasks
import simplepeft.train.train

simplepeft.train.train.ACCUMULATION_STEPS = 32

BATCH_SIZE = 8
BASE_MODEL = "mDeBERTa-v3-llm-tasks-classification"
PEFT_MODEL = "mDeBERTa-v3-llm-tasks-classification"
TASK = Tasks.TEXT_CLASSIFICATION
LR = 1e-5

# get all the data
ds = datasets.load_dataset("flozi00/LLM-Task-Classification", split="train")
texts = ds["text"]
named_labels = ds["named_labels"]

ds = datasets.load_dataset("flozi00/classify-llm-tasks-german", split="train")
texts += ds["input"]
named_labels += ds["output"]

ds = datasets.Dataset.from_dict({"text": texts, "named_labels": named_labels})

labels = ds.unique("named_labels")
labels.sort()

LABELS_TO_IDS = {label: i for i, label in enumerate(labels)}
IDS_TO_LABELS = {i: label for i, label in enumerate(labels)}

print(LABELS_TO_IDS)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABELS_TO_IDS.keys()),
    id2label=IDS_TO_LABELS,
    label2id=LABELS_TO_IDS,
    ignore_mismatched_sizes=True,
)

model = model.train()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


# get the dataloader and define config for data loading and transformation
dloader = get_dataloader(
    task=TASK,  # type: ignore
    processor=tokenizer,
    datas=ds,
    BATCH_SIZE=BATCH_SIZE,
    text_key="text",
    label_key="named_labels",
    label_to_id=LABELS_TO_IDS,
    max_input_length=512,
)

# start training
simplepeft.train.train.start_training(
    model=model,
    processor=tokenizer,
    dloader=dloader,
    PEFT_MODEL=PEFT_MODEL,
    LR=LR,
    num_epochs=10,
)