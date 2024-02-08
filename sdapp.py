from atra.gradio_utils.sd import build_diffusion_ui
import gradio as gr
from fastapi import FastAPI

app = FastAPI()


blocks = build_diffusion_ui()

app = gr.mount_gradio_app(app, blocks, path="/")
