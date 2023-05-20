import os
import threading

from atra.gradio_utils import build_gradio
from atra.text_utils.question_answering import answer_question
from atra.text_utils.summarization import summarize
from atra.text_utils.translation import translate

ui = build_gradio()


class BackgroundTasks(threading.Thread):
    def run(self, *args, **kwargs):
        import numpy as np
        from atra.audio_utils.asr import speech_recognition
        from atra.datastore import (
            get_tasks_queue,
            set_transkript,
            QueueData
        )

        while True:
            task = get_tasks_queue()
            if isinstance(task, QueueData):
                bytes_data = task.file_object
                task_metas = task.metas.split(",")
                result = None
                if task_metas[-1] == "translation":
                    input_text = bytes_data.decode("utf-8")
                    input_lang = task_metas[0]
                    target_lang = task_metas[1]
                    result = translate(input_text, input_lang, target_lang)
                elif task_metas[-1] == "summarization":
                    input_text = bytes_data.decode("utf-8")
                    input_lang = task_metas[0]
                    target_lang = task_metas[1]
                    result = summarize(input_text, input_lang)
                elif task_metas[-1] == "question-answering":
                    input_text = bytes_data.decode("utf-8")
                    question = task_metas[0]
                    input_lang = task_metas[1]
                    result = answer_question(input_text, question, input_lang)
                elif task_metas[-1] == "asr":
                    array = np.frombuffer(bytes_data, dtype=np.float32)
                    lang = task_metas[2]
                    result = speech_recognition(
                        data=array,
                        language=lang,
                    )
                if result is not None:
                    set_transkript(hs=task.hash, transcription=result)


if __name__ == "__main__":
    worker = os.getenv("RUNWORKER", "true").lower() == "true"
    if worker is True:
        print("Starting worker")
        t = BackgroundTasks()
        t.start()

    auth_name = os.getenv("AUTH_NAME", None)
    auth_password = os.getenv("AUTH_PASSWORD", None)

    ui.launch(
        enable_queue=False,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        auth=(auth_name, auth_password) if 
        auth_name is not None and auth_password is not None 
        else None,
    )
