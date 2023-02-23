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
from aaas.gradio_utils import add_vad_chunks
from aaas.statics import TO_VAD, TO_OCR
import numpy as np

from aaas.gradio_utils import build_gradio
from fastapi import FastAPI, staticfiles
import uvicorn
import gradio as gr

app = FastAPI()
ui = build_gradio()

app.mount("/gradio", gr.routes.App.create_app(ui))
app.mount(
    "/",
    staticfiles.StaticFiles(directory="client/build/web", html=True),
    name="static",
)


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

    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)), log_level="debug"
    )
