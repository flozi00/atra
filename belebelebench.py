from huggingface_hub import InferenceClient
from datasets import load_dataset
import time

llm = InferenceClient("meta-llama/Llama-2-70b-chat-hf")

ds = load_dataset(
    "facebook/belebele",
    "deu_Latn",
    split="test",
)


def get_answer(prompt: str) -> str:
    return llm.text_generation(
        prompt=prompt,
        temperature=0.1,
        stop_sequences=["\n"],
        max_new_tokens=3,
    ).strip()


runs = 0
correct = 0

for entry in ds:
    ERROR = True
    flores_passage = entry["flores_passage"]
    question = entry["question"]
    mc_answer1 = entry["mc_answer1"]
    mc_answer2 = entry["mc_answer2"]
    mc_answer3 = entry["mc_answer3"]
    mc_answer4 = entry["mc_answer4"]
    correct_answer_num = entry["correct_answer_num"]

    PROMPT = f"""Aufgabe: Gegeben ist ein Text in deutscher Sprache und eine Frage auf Deutsch.
Die Antwort auf die Frage ist die Nummer einer der vier vorgegebenen Antworten.

Antwortmöglichkeiten:
1. {mc_answer1}
2. {mc_answer2}
3. {mc_answer3}
4. {mc_answer4}

Text: {flores_passage}

Frage: {question}

Antwort:"""

    while ERROR:
        try:
            answer = get_answer(PROMPT)
            ERROR = False
        except Exception as e:
            print(e)
            time.sleep(5)

    if str(correct_answer_num) in answer:
        correct += 1
    runs += 1

    print(f"Correct: {correct}/{runs} ({correct/runs*100:.2f}%)")
