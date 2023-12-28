from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    UniPCMultistepScheduler,
)

import torch
from atra.utilities.stats import timeit
import time
import json
import diffusers.pipelines.stable_diffusion_xl.watermark
from atra.image_utils.free_lunch_utils import (
    register_free_crossattn_upblock2d,
)
import gradio as gr
import pathlib
import torch._inductor


def apply_watermark_dummy(self, images: torch.FloatTensor):
    return images


diffusers.pipelines.stable_diffusion_xl.watermark.StableDiffusionXLWatermarker.apply_watermark = (
    apply_watermark_dummy
)

GPU_AVAILABLE = torch.cuda.is_available()

_images_per_prompt = 2
INFER_STEPS = 20
GPU_ID = 0
POWER = 450 if GPU_AVAILABLE else 100
if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(GPU_ID)
else:
    GPU_NAME = "CPU"
if "H100" in GPU_NAME:
    POWER = 310
elif "A6000" in GPU_NAME:
    POWER = 300
elif "RTX 6000" in GPU_NAME:
    POWER = 240
elif "L40" in GPU_NAME:
    POWER = 350


diffusion_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stablediffusionapi/opendalle",
    torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
    use_safetensors=False,
)

if GPU_AVAILABLE:
    diffusion_pipe = diffusion_pipe.to(f"cuda:{GPU_ID}")

diffusion_pipe.unet.to(memory_format=torch.channels_last)
diffusion_pipe.vae.to(memory_format=torch.channels_last)

diffusion_pipe.fuse_qkv_projections()

# change scheduler
diffusion_pipe.scheduler = UniPCMultistepScheduler.from_config(
    diffusion_pipe.scheduler.config
)


@timeit
def generate_images(
    prompt: str,
    negatives: str = "",
    progress=gr.Progress(track_tqdm=True),
):
    TIME_LOG = {"GPU Power insert in W": POWER}

    if negatives is None:
        negatives = ""

    start_time = time.time()
    register_free_crossattn_upblock2d(
        diffusion_pipe,
        b1=1.3,
        b2=1.4,
        s1=0.9,
        s2=0.2,
    )
    with torch.inference_mode():
        image = diffusion_pipe(
            prompt=prompt,
            negative_prompt=negatives,
            num_inference_steps=INFER_STEPS,
            num_images_per_prompt=_images_per_prompt,
        ).images

    consumed_time = time.time() - start_time
    TIME_LOG["Time in seconds"] = consumed_time
    TIME_LOG["Comsumed Watt hours"] = consumed_time * POWER / 3600
    TIME_LOG["Energy costs in cent"] = TIME_LOG["Comsumed Watt hours"] * 40 / 1000
    TIME_LOG["Device Name"] = GPU_NAME

    MD = json.dumps(TIME_LOG, indent=4)
    MD = "```json\n" + MD + "\n```"

    paths = []
    for x in range(len(image)):
        image[x].save("output_image.jpg", "JPEG", optimize=True)
        paths.append(pathlib.Path(f"output_image_{x}.jpg"))

    return image, MD
