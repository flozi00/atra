from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverSinglestepScheduler,
)
import torch
from atra.utils import timeit

pipe = None
refiner = None

import diffusers.pipelines.stable_diffusion_xl.watermark


def apply_watermark_dummy(self, images: torch.FloatTensor):
    return images


diffusers.pipelines.stable_diffusion_xl.watermark.StableDiffusionXLWatermarker.apply_watermark = (
    apply_watermark_dummy
)


def get_pipes():
    global pipe, refiner
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe.to("cuda")
    refiner.to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    refiner.enable_xformers_memory_efficient_attention()

    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)


@timeit
def generate_images(prompt: str, negatives: str = ""):
    n_steps = 15
    high_noise_frac = 0.8

    if pipe is None:
        get_pipes()

    if negatives is None:
        negatives = ""

    image = pipe(
        prompt=prompt,
        negative_prompt=negatives,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        negative_prompt=negatives,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    return image
