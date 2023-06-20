import gradio as gr
from atra.model_utils.model_utils import get_model_and_processor


def generate_images(prompt: str, progress = gr.Progress()):
    pipe, processor = get_model_and_processor(
        lang="photo-real", task="diffusion", progress=progress
    )

    progress.__call__(progress=0.8, desc="Generating images")
    image = pipe(prompt, num_images_per_prompt=1).images[0]

    return image
