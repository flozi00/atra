from atra.model_utils.model_utils import get_model_and_processor

models = ["openjourney", "photo-real"]


def generate_images(prompt: str, num_inference_steps: int):
    images = []
    for mode in models:
        pipeline, processor = get_model_and_processor(lang=mode, task="diffusion")

        image = pipeline(
            prompt, num_images_per_prompt=4, num_inference_steps=num_inference_steps
        ).images
        for i in image:
            images.append(i)

        yield images
