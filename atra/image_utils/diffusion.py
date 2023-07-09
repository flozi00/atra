from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-0.9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.unet = torch.compile(
    pipe.unet, mode="reduce-overhead", backend="onnxrt", fullgraph=True
)
refiner.unet = torch.compile(
    refiner.unet, mode="reduce-overhead", backend="onnxrt", fullgraph=True
)


def generate_images(prompt: str, negatives: str = ""):
    if negatives is None:
        negatives = ""
    image = pipe(prompt=prompt, negative_prompt=negatives, output_type="latent").images[
        0
    ]
    image = refiner(
        prompt=prompt, negative_prompt=negatives, image=image[None, :]
    ).images[0]

    return image
