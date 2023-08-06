import pathlib
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverSinglestepScheduler,
    AutoencoderTiny,
    EulerAncestralDiscreteScheduler
)
from diffusers.models.attention_processor import AttnProcessor2_0

import torch
from atra.utils import timeit, ttl_cache
import time
import subprocess
import json
import diffusers.pipelines.stable_diffusion_xl.watermark


def apply_watermark_dummy(self, images: torch.FloatTensor):
    return images


diffusers.pipelines.stable_diffusion_xl.watermark.StableDiffusionXLWatermarker.apply_watermark = (
    apply_watermark_dummy
)

TEMP_LIMIT = 65

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

diffusion_pipe = StableDiffusionXLPipeline.from_single_file("https://huggingface.co/jayparmr/DreamShaper_XL1_0_Alpha2/blob/main/dreamshaperXL10.safetensors", torch_dtype=torch.float16)
diffusion_pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
diffusion_pipe.unet.set_attn_processor(AttnProcessor2_0())

diffusion_pipe.vae = torch.compile(diffusion_pipe.vae, mode="reduce-overhead", fullgraph=True)
diffusion_pipe = diffusion_pipe.to("cuda")
diffusion_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        diffusion_pipe.scheduler.config
    )

@timeit
def generate_images(prompt: str, negatives: str = ""):
    TIME_LOG = {"gpu-power": POWER}

    for pattern in BAD_PATTERNS:
        if pattern in prompt:
            raise "NSFW prompt not allowed"

    if negatives is None:
        negatives = ""

    start_time = time.time()
    with torch.inference_mode():
        image = diffusion_pipe(
            prompt=prompt,
            negative_prompt=negatives,
            num_inference_steps=60,
        ).images[0]
    TIME_LOG["base-inference"] = time.time() - start_time
    TIME_LOG["watt-seconds"] = TIME_LOG["base-inference"] * POWER

    TIME_LOG["watt-hours"] = TIME_LOG["watt-seconds"] / 3600

    TIME_LOG["energy-costs-euro"] = TIME_LOG["watt-hours"] / 1000 * 0.4
    TIME_LOG["energy-costs-cent"] = TIME_LOG["energy-costs-euro"] * 100

    image.save("output_image.jpg", "JPEG")
    image = pathlib.Path("output_image.jpg")
    return image, TIME_LOG
