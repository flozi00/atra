import os
import threading
from aaas.datastore import remove_data_from_hash

from aaas.gradio_utils import build_gradio

ui = build_gradio()

from transformers.tools import TranslationTool

translator = TranslationTool()


class BackgroundTasks(threading.Thread):
    def run(self, *args, **kwargs):
        import numpy as np
        from aaas.audio_utils.asr import inference_asr
        from aaas.datastore import (
            get_data_from_hash,
            get_tasks_queue,
            set_in_progress,
            set_transkript,
        )
        import time

        while True:
            task = get_tasks_queue()
            if task is not False:
                bytes_data = get_data_from_hash(task.hash)
                set_in_progress(task.hash)
                task_metas = task.metas.split(",")
                if task_metas[-1] == "translation":
                    input_text = bytes_data.decode("utf-8")
                    input_lang = task_metas[0]
                    target_lang = task_metas[1]
                    result = translator(input_text, input_lang, target_lang)
                elif task_metas[-1] == "asr":
                    array = np.frombuffer(bytes_data, dtype=np.float32)
                    result, lang = inference_asr(
                        data=array,
                        model_config=None,
                    )
                set_transkript(task.hash, result, from_queue=True)
                remove_data_from_hash(task.hash)
            else:
                time.sleep(1)


if __name__ == "__main__":
    worker = os.getenv("RUNWORKER", "true").lower() == "true"
    if worker is True:
        print("Starting worker")
        t = BackgroundTasks()
        t.start()

    ui.launch(
        enable_queue=False,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
    )
