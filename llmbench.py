from huggingface_hub import InferenceClient
from datasets import load_dataset
import time

llm = InferenceClient("meta-llama/Llama-2-70b-chat-hf")


def get_answer(prompt: str) -> str:
    ERROR = True
    while ERROR:
        try:
            answer = llm.text_generation(
                prompt=prompt,
                temperature=0.1,
                stop_sequences=["\n"],
                max_new_tokens=128,
            ).strip()
            ERROR = False
        except Exception as e:
            print(e)
            time.sleep(5)
    return answer


runs = 0
correct = 0


#############################################
# Reading Comprehension
# Choose the correct answer to the question
#############################################
def belebele_benchmark() -> None:
    global runs, correct
    ds = load_dataset(
        "facebook/belebele",
        "deu_Latn",
        split="test",
    )

    for entry in ds:
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

        answer = get_answer(PROMPT)
        if str(correct_answer_num) in answer:
            correct += 1
        runs += 1

        print(f"Correct: {correct}/{runs} ({correct/runs*100:.2f}%)")


#############################################
# Question Answering
# Answer the question, extract the answer from the text
#############################################
def germanquad_benchmark() -> None:
    global runs, correct
    ds = load_dataset("deepset/germanquad", split="test")
    for entry in ds:
        context = entry["context"]
        question = entry["question"]
        answers = entry["answers"]["text"]

        PROMPT = f"""Aufgabe: Gegeben ist ein Text in deutscher Sprache und eine Frage auf Deutsch.
Die Antwort auf die Frage wird aus dem Text extrahiert.

Text: {context}

Frage: {question}

Antwort:"""

        answer = get_answer(PROMPT)

        for ans in answers:
            if ans.lower() in answer.lower() or answer.lower() in ans.lower():
                correct += 1
                break
        runs += 1

        print(f"Correct: {correct}/{runs} ({correct/runs*100:.2f}%)")


# belebele_benchmark()
germanquad_benchmark()
