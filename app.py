import os
import time
import threading
from aaas.audio_utils.asr import inference_asr
from aaas.vision_utils.ocr import inference_ocr
from aaas.datastore import (
    get_tasks_queue,
    set_in_progress,
    set_transkript,
)
from aaas.text_utils import translate
from aaas.gradio_utils import add_vad_chunks
from aaas.statics import LANG_MAPPING, TO_VAD, TO_OCR
import numpy as np


class BackgroundTasks(threading.Thread):
    def run(self, *args, **kwargs):
        while True:
            task = get_tasks_queue()
            if task is not False:
                set_in_progress(task.hash)
                if task.metas == TO_VAD:
                    array = np.frombuffer(task.data, dtype=np.float32)
                    result = add_vad_chunks(
                        audio=array,
                        main_lang=task.langs.split(",")[0],
                        model_config=task.model_config,
                    )
                elif task.metas == TO_OCR:
                    with open("dummy.png", "wb") as f:
                        f.write(task.data)
                    result = inference_ocr(
                        data="dummy.png",
                        mode=task.langs.split(",")[0],
                        config=task.model_config,
                    )
                else:
                    array = np.frombuffer(task.data, dtype=np.float32)
                    result = inference_asr(
                        data=array,
                        main_lang=task.langs.split(",")[0],
                        model_config=task.model_config,
                    )
                set_transkript(task.hash, result)
            else:
                time.sleep(3)


if __name__ == "__main__":
    t = BackgroundTasks()
    t.start()
    from aaas.gradio_utils import build_gradio

    ui = build_gradio()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        max_threads=32,
    )
