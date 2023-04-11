import os
import threading
import time

import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI, staticfiles

from aaas.audio_utils.asr import inference_asr
from aaas.datastore import (
    get_data_from_hash,
    get_tasks_queue,
    set_in_progress,
    set_transkript,
)
from aaas.gradio_utils import build_gradio

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
            task, is_reclamation = get_tasks_queue()
            if task is not False:
                bytes_data = get_data_from_hash(task.hash)
                set_in_progress(task.hash)
                array = np.frombuffer(bytes_data, dtype=np.float32)
                result, lang = inference_asr(
                    data=array,
                    model_config=task.model_config,
                    is_reclamation=is_reclamation,
                )
                set_transkript(task.hash, result, from_queue=True, lang=lang)
            else:
                time.sleep(1)


if __name__ == "__main__":
    t = BackgroundTasks()
    t.start()

    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)), log_level="info"
    )
