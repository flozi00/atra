import time
from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from atra.image_utils.diffusion import generate_images
import gradio as gr
from gradio_client import Client
import os

IMAGE_BACKENDS = os.getenv("SD")
if IMAGE_BACKENDS is not None:
    IMAGE_BACKENDS = IMAGE_BACKENDS.split(",")
else:
    IMAGE_BACKENDS = []

CLIENTS = [Client(src=backend) for backend in IMAGE_BACKENDS]


def use_diffusion_ui(prompt, negatives):
    jobs = [client.submit(prompt, negatives, api_name="/sd") for client in CLIENTS]
    results = []
    running_job = True

    while running_job:
        results = []
        running_job = False
        for job in jobs:
            if not job.done():
                running_job = True
            else:
                res = job.result()
                results.append(res)
                yield results


def build_diffusion_ui():
    ui = gr.Blocks(css=GLOBAL_CSS)
    with ui:
        GET_GLOBAL_HEADER()
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                negatives = gr.Textbox(label="Negative Prompt")
            images = gr.Image() if len(CLIENTS) == 0 else gr.Gallery()

        prompt.submit(
            generate_images if len(CLIENTS) == 0 else use_diffusion_ui,
            inputs=[prompt, negatives],
            outputs=images,
            api_name="sd",
        )
        negatives.submit(
            generate_images if len(CLIENTS) == 0 else use_diffusion_ui,
            inputs=[prompt, negatives],
            outputs=images,
        )

    ui.queue(concurrency_count=1, api_open=False)
    ui.launch(server_port=7861, **launch_args)
