from atra.model_utils.model_utils import get_model_and_processor


def generate_images(prompt: str, num_inference_steps: int):
    pipeline, processor = get_model_and_processor(lang="openjourney", task="diffusion")

    image = pipeline(
        prompt,
        num_images_per_prompt=1,
        num_inference_steps=int(num_inference_steps),
    ).images

    return image[0]
