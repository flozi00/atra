from diffusers import (
    DiffusionPipeline,
)
import torch
from atra.utils import timeit

pipe = None
refiner = None


def get_pipes():
    global pipe, refiner
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-0.9",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-0.9",
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

    pipe.unet = torch.compile(
        pipe.unet, backend="inductor", mode="reduce-overhead", fullgraph=True
    )
    pipe.vae = torch.compile(
        pipe.vae, backend="onnxrt", mode="reduce-overhead", fullgraph=True
    )

    refiner.unet = torch.compile(
        refiner.unet, backend="inductor", mode="reduce-overhead", fullgraph=True
    )
    refiner.vae = torch.compile(
        refiner.vae, backend="onnxrt", mode="reduce-overhead", fullgraph=True
    )

    # pipe.enable_model_cpu_offload()
    # refiner.enable_model_cpu_offload()


@timeit
def generate_images(prompt: str, negatives: str = ""):
    n_steps = 40
    high_noise_frac = 0.7

    if pipe is None:
        get_pipes()

    if negatives is None:
        negatives = ""
    image = pipe(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    return image
