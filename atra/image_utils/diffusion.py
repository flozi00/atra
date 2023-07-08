from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.enable_model_cpu_offload()

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-0.9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.enable_model_cpu_offload()


def generate_images(prompt: str):
    image = pipe(prompt=prompt, output_type="latent").images[0]
    image = refiner(prompt=prompt, image=image[None, :]).images[0]

    return image
