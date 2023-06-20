import gradio as gr
from atra.model_utils.model_utils import get_model_and_processor

def generate_images(prompt: str, num_inference_steps: int, progress = gr.Progress()):
    pipeline, processor = get_model_and_processor(
        lang="photo-real", task="diffusion", progress=progress
    )

    image = pipeline(prompt, num_images_per_prompt=1, num_inference_steps=num_inference_steps).images[0]
    return image
