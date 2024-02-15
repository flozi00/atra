from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
import gradio as gr
import base64
import io
import numpy as np
from pytriton.client import ModelClient
from PIL import Image

IMAGE_FORMAT = "JPEG"


def _encode_image_to_base64(image: Image.Image):
    raw_bytes = io.BytesIO()
    image.save(raw_bytes, IMAGE_FORMAT)
    raw_bytes.seek(0)  # return to the start of the buffer
    return base64.b64encode(raw_bytes.read())

def infer_client(prompt: str, img_size_height: int, image_size_width: int, face_image = None):
    files = []
    with ModelClient("localhost", "SDXL") as client:
        img_size_height = np.array([img_size_height])
        image_size_width = np.array([image_size_width])
        prompt = np.array([prompt])
        prompt = np.char.encode(prompt, "utf-8")
        if face_image is not None:
            face_image = Image.open(face_image)
            face_image = _encode_image_to_base64(face_image)
            face_image = np.array([face_image])
            result_dict = client.infer_sample(prompt=prompt, img_size_height=img_size_height, image_size_width=image_size_width, image=face_image)
        else:
            result_dict = client.infer_sample(prompt=prompt, img_size_height=img_size_height, image_size_width=image_size_width)

        image = result_dict["image"][0]
        file_path = f"output_image.jpeg"
        msg = base64.b64decode(image)
        buffer = io.BytesIO(msg)
        image = Image.open(buffer)
        image.save(file_path, "JPEG")
        files.append(file_path)

        prompt = result_dict["prompt"][0]
        prompts = np.char.decode(prompt, "utf-8").item()

    return files, prompts


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
            face_image = gr.Image(label="Face Image", type="filepath")
            height = gr.Slider(
                minimum=512,
                maximum=1024,
                step=32,
                value=512,
                label="Height",
            )
            width = gr.Slider(
                minimum=512,
                maximum=1024,
                step=32,
                value=512,
                label="Width",
            )

        prompt.submit(
            infer_client,
            inputs=[prompt, height, width, face_image],
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

        gr.Textbox(value="the actual model used is: dataautogpt3/ProteusV0.3")

    ui.queue(api_open=False)

    ui.launch(**launch_args)
