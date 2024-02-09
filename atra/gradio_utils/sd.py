from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
import gradio as gr
import base64
import io
import numpy as np
from pytriton.client import ModelClient
from PIL import Image


def infer_client(prompt: str, img_size: int):
    files = []
    with ModelClient("localhost", "SDXL") as client:
        img_size = np.array([[img_size]])
        prompt = np.array([[prompt]])
        prompt = np.char.encode(prompt, "utf-8")

        result_dict = client.infer_batch(prompt=prompt, img_size=img_size)

        for idx, image in enumerate(result_dict["image"]):
            file_path = f"output_image_{idx}.jpeg"
            msg = base64.b64decode(image[0])
            buffer = io.BytesIO(msg)
            image = Image.open(buffer)
            image.save(file_path, "JPEG")
            files.append(file_path)

        prompt = result_dict["prompt"]
        prompts = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in prompt]

    return files, prompts[0]


def build_diffusion_ui() -> None:
    ui = gr.Blocks(css=GLOBAL_CSS)
    with ui:
        with gr.Row():
            GET_GLOBAL_HEADER()
        with gr.Column():
            images = gr.Gallery(label="Image")

        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt", info="Prompt of what you want to see", interactive=True
            )
            height = gr.Slider(
                minimum=512,
                maximum=1024,
                step=32,
                value=512,
                label="Height",
            )

        prompt.submit(
            infer_client,
            inputs=[prompt, height],
            outputs=[images, prompt],
        )

        gr.Examples(
            [
                ["cyborg style, golden retriever"],
                [
                    "High nation-geographic symmetrical close-up portrait shoot in green jungle of an expressive lizard, anamorphic lens, ultra-realistic, hyper-detailed, green-core, jungle-core",
                ],
                ["Glowing jellyfish floating through a foggy forest at twilight", ""],
                [
                    "Elegant lavender garnish cocktail idea, cocktail glass, realistic, sharp focus, 8k high definition",
                ],
                [
                    "General Artificial Intelligence in data center, futuristic concept art, 3d rendering",
                ],
            ],
            inputs=[prompt],
        )

        gr.Textbox(value="the actual model used is: dataautogpt3/ProteusV0.2")

    ui.queue(api_open=False)

    ui.launch(**launch_args)
