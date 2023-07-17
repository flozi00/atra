import time
from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from atra.image_utils.diffusion import generate_images
import gradio as gr
from gradio_client import Client

client = Client(src="https://chat.atra.ai")


def use_diffusion_ui(prompt, negatives):
    job = client.submit(3, api_name="/sd")
    while not job.done():
        time.sleep(0.1)

    return job.result()


def build_diffusion_ui():
    ui = gr.Blocks(css=GLOBAL_CSS)
    with ui:
        GET_GLOBAL_HEADER()
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                negatives = gr.Textbox(label="Negative Prompt")
            images = gr.Image()

        prompt.submit(
            generate_images, inputs=[prompt, negatives], outputs=images, api_name="sd"
        )
        negatives.submit(generate_images, inputs=[prompt, negatives], outputs=images)

    ui.queue(concurrency_count=1, api_open=False)
    ui.launch(server_port=7861, **launch_args)
