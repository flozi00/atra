from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from atra.image_utils.diffusion import generate_images
import gradio as gr


def build_diffusion_ui():
    ui = gr.Blocks(css=GLOBAL_CSS)
    with ui:
        GET_GLOBAL_HEADER()
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                negatives = gr.Textbox(label="Negative Prompt")
            images = gr.Image()

        prompt.submit(generate_images, inputs=[prompt, negatives], outputs=images)
        negatives.submit(generate_images, inputs=[prompt, negatives], outputs=images)

    ui.queue(concurrency_count=1, api_open=False)
    ui.launch(server_port=7861, **launch_args)
