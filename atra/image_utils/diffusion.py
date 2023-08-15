import pathlib
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
import torch
from atra.utils import timeit
import time
import subprocess
import json
import diffusers.pipelines.stable_diffusion_xl.watermark


def apply_watermark_dummy(self, images: torch.FloatTensor):
    return images


diffusers.pipelines.stable_diffusion_xl.watermark.StableDiffusionXLWatermarker.apply_watermark = (
    apply_watermark_dummy
)

high_noise_frac = 0.8
INFER_STEPS = 60

BAD_PATTERNS = [
    "nude",
    "naked",
    "nacked",
    "porn",
    "undressed",
    "sex",
    "erotic",
    "pornographic",
    "vulgar",
    "hentai",
    "nackt",
]


subprocess = subprocess.Popen("gpustat -P --json", shell=True, stdout=subprocess.PIPE)
subprocess_return = subprocess.stdout.read()
POWER = json.loads(subprocess_return)["gpus"][0]["enforced.power.limit"]

diffusion_pipe = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/jayparmr/DreamShaper_XL1_0_Alpha2/blob/main/dreamshaperXL10.safetensors",
    torch_dtype=torch.float16,
)
# diffusion_pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
diffusion_pipe.unet.set_attn_processor(AttnProcessor2_0())
diffusion_pipe.vae = torch.compile(
    diffusion_pipe.vae, mode="reduce-overhead", fullgraph=True
)
diffusion_pipe = diffusion_pipe.to("cuda")
diffusion_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    diffusion_pipe.scheduler.config
)

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=diffusion_pipe.text_encoder_2,
    vae=diffusion_pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.scheduler = EulerAncestralDiscreteScheduler.from_config(
    refiner.scheduler.config
)
refiner.unet.set_attn_processor(AttnProcessor2_0())
# refiner.vae = torch.compile(refiner.vae, mode="reduce-overhead", fullgraph=True)
refiner.to("cuda")


@timeit
def generate_images(prompt: str, negatives: str = ""):
    TIME_LOG = {"GPU Power insert in W": POWER}

    # for pattern in BAD_PATTERNS:
    #    if pattern in prompt:
    #        raise gr.Error("NSFW prompt not allowed")
    # raise "NSFW prompt not allowed"

    if negatives is None:
        negatives = ""

    start_time = time.time()
    with torch.inference_mode():
        image = diffusion_pipe(
            prompt=prompt,
            negative_prompt=negatives,
            num_inference_steps=INFER_STEPS,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images[0]
        image = refiner(
            prompt=prompt,
            num_inference_steps=INFER_STEPS,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

    TIME_LOG["Comsumed Watt hours"] = (time.time() - start_time) * POWER / 3600
    TIME_LOG["Energy costs in cent"] = TIME_LOG["Comsumed Watt hours"] * 40 / 1000

    image.save("output_image.jpg", "JPEG")
    image = pathlib.Path("output_image.jpg")
    return image, TIME_LOG
