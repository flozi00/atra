from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from atra.image_utils.diffusion import generate_images as local_generator
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
    jobs = [client.submit(prompt, negatives, fn_index=0) for client in CLIENTS]
    results = []
    for c in CLIENTS:
        results.append(None)
        results.append(None)
    running_job = True

    while running_job:
        running_job = False
        for job_index in range(len(jobs)):
            job = jobs[job_index]
            if not job.done():
                running_job = True
            else:
                img, log = job.result()
                results[job_index*2] = img
                results[job_index*2+1] = log
                
                yield results

if len(CLIENTS) == 0:
    generate_images = local_generator
else:
    generate_images = use_diffusion_ui

def build_diffusion_ui():
    ui = gr.Blocks(css=GLOBAL_CSS)
    with ui:
        GET_GLOBAL_HEADER()
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt", info="Prompt of what you want to see"
                )
                negatives = gr.Textbox(
                    label="Negative Prompt",
                    info="Prompt describing what you dont want to see, useful for refining image",
                )
            
            if len(CLIENTS) == 0:
                _boxes = []
                with gr.Column():
                    _boxes.append(gr.Image())
                    _boxes.append(gr.JSON())
            else:
                _boxes = []
                for c in CLIENTS:
                    with gr.Column():
                        _boxes.append(gr.Image())
                        _boxes.append(gr.JSON())


        prompt.submit(
            generate_images,
            inputs=[prompt, negatives],
            outputs=_boxes,
        )
        negatives.submit(
            generate_images,
            inputs=[prompt, negatives],
            outputs=_boxes,
        )

        gr.Examples(
            [
                [
                    "A photo of A majestic lion jumping from a big stone at night",
                ],
                [
                    "Aerial photography of a winding river through autumn forests, with vibrant red and orange foliage",
                ],
                [
                    "interior design, open plan, kitchen and living room, modular furniture with cotton textiles, wooden floor, high ceiling, large steel windows viewing a city",
                ],
                [
                    "High nation-geographic symmetrical close-up portrait shoot in green jungle of an expressive lizard, anamorphic lens, ultra-realistic, hyper-detailed, green-core, jungle-core"
                ],
                ["photo of romantic couple walking on beach while sunset"],
                ["Glowing jellyfish floating through a foggy forest at twilight"],
                [
                    "Skeleton man going on an adventure in the foggy hills of Ireland wearing a cape"
                ],
                [
                    "Elegant lavender garnish cocktail idea, cocktail glass, realistic, sharp focus, 8k high definition"
                ],
            ],
            inputs=[prompt],
        )

    ui.queue(concurrency_count=1, api_open=False)
    ui.launch(server_port=7861, **launch_args)
