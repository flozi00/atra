from atra.gradio_utils.asr import build_asr_ui

import gradio as gr
from fastapi import FastAPI

app = FastAPI()


blocks = build_asr_ui()

app = gr.mount_gradio_app(app, blocks, path="/")
