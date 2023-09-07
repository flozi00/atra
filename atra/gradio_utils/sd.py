from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
import gradio as gr
import json
import pandas as pd
from gradio_client import Client
import os

IMAGE_BACKENDS = os.getenv("SD")
if IMAGE_BACKENDS is not None:
    IMAGE_BACKENDS = IMAGE_BACKENDS.split(",")
else:
    IMAGE_BACKENDS = []
    from atra.image_utils.diffusion import generate_images as local_generator


CLIENTS = [Client(src=backend) for backend in IMAGE_BACKENDS]

def calculate_efficiency(gpu_name, watthours, time_in_seconds) -> int:
    MAX_PRECISION = 32
    if "H100" in gpu_name:
        RAM = 80
        MIN_PRECISION = 8
    elif "A6000" in gpu_name:
        RAM = 48
        MIN_PRECISION = 16
    elif "RTX 6000" in gpu_name:
        RAM = 24
        MIN_PRECISION = 16
    else:
        RAM = 24
        MIN_PRECISION = 16
    
    efficiency = (1/watthours) * RAM * (MAX_PRECISION / MIN_PRECISION) * (1 / time_in_seconds)

    return int(efficiency)

def use_diffusion_ui(prompt, negatives):
    jobs = [client.submit(prompt, negatives, fn_index=0) for client in CLIENTS]
    results = []
    for c in CLIENTS:
        results.append(None)
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
                results[job_index * 2] = img
                results[job_index * 2 + 1] = log

                yield results
    
    scores = []
    names = []
    for i in range(len(results)):
        if i % 2 == 1:
            data = results[i].replace("```json\n", "")
            data = data.replace("\n```", "")
            data = json.loads(data)
            score = calculate_efficiency(data["Device Name"], data["Comsumed Watt hours"], data["Time in seconds"])
            scores.append(score)
            names.append(data["Device Name"])

    baseline_score = min(scores)
    scores = [1 / baseline_score * score for score in scores]

    simple = pd.DataFrame(
        {
            "GPU": names,
            "Score": scores,
        }
    )


    results[-1] = gr.BarPlot.update(simple,
                x="GPU",
                y="Score",
                title="Efficiency Score using the worst GPU as baseline (higher is better)",
                width=300,
                min_width=300,
                vertical=False,
    )

    yield results


if len(CLIENTS) == 0:
    generate_images = local_generator
else:
    generate_images = use_diffusion_ui


def build_diffusion_ui() -> None:
    ui = gr.Blocks(css=GLOBAL_CSS)
    with ui:
        with gr.Row():
            GET_GLOBAL_HEADER()
            if len(CLIENTS) > 0:
                gr.Markdown("# GPU Vergleich")
        with gr.Row():
            if len(CLIENTS) == 0:
                _boxes = []
                with gr.Column():
                    _boxes.append(gr.Image())
                    _boxes.append(gr.Markdown())
            else:
                _boxes = []
                for c in range(len(CLIENTS)):
                    with gr.Column():
                        gr.Markdown("GPU " + str(c))
                        _boxes.append(gr.Image())
                        _boxes.append(gr.Markdown())
                _boxes.append(gr.BarPlot())
        
        with gr.Column():
            if len(CLIENTS) > 0:
                gr.Markdown("### Prompt")
            prompt = gr.Textbox(
                label="Prompt", info="Prompt of what you want to see"
            )
            negatives = gr.Textbox(
                label="Negative Prompt",
                info="Prompt describing what you dont want to see, useful for refining image",
            )

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
                [
                    "General Artificial Intelligence in data center, futuristic concept art, 3d rendering"
                ],
            ],
            inputs=[prompt],
        )

    ui.queue(concurrency_count=1, api_open=False)
    ui.launch(**launch_args)