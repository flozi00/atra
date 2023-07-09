from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from diffusers import DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-0.9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")
refiner.to("cuda")


pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)


def generate_images(prompt: str, negatives: str = ""):
    if negatives is None:
        negatives = ""
    image = pipe(
        prompt=prompt,
        negative_prompt=negatives,
        output_type="latent",
        num_images_per_prompt=1,
        num_inference_steps=30,
    ).images[0]
    image = refiner(
        prompt=prompt, negative_prompt=negatives, image=image[None, :]
    ).images[0]

    return image
