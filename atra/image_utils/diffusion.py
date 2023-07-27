from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
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

    pipe.to("cuda")

    pipe.enable_xformers_memory_efficient_attention()


@timeit
def generate_images(prompt: str, negatives: str = "", mode: str = "prototyping"):
    if pipe is None:
        get_pipes()

    if negatives is None:
        negatives = ""

    if mode == "prototyping":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        n_steps = 20
    else:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        n_steps = 50

    image = pipe(
        prompt=prompt,
        negative_prompt=negatives,
        num_inference_steps=n_steps,
    ).images[0]

    return image
