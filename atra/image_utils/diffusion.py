from diffusers import (
    StableDiffusionXLPipeline,
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
from DeepCache import DeepCacheSDHelper

# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True


def apply_watermark_dummy(self, images: torch.FloatTensor):
    return images


diffusers.pipelines.stable_diffusion_xl.watermark.StableDiffusionXLWatermarker.apply_watermark = (
    apply_watermark_dummy
)

GPU_AVAILABLE = torch.cuda.is_available()

_images_per_prompt = 1
INFER_STEPS = 80
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
    "dataautogpt3/ProteusV0.2",
    torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
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

# diffusion_pipe.unet = torch.compile(
#    diffusion_pipe.unet, mode="reduce-overhead", fullgraph=True
# )
# diffusion_pipe.vae.decode = torch.compile(
#    diffusion_pipe.vae.decode, mode="reduce-overhead", fullgraph=True
# )

helper = DeepCacheSDHelper(pipe=diffusion_pipe)
helper.set_params(
    cache_interval=6,
    cache_branch_id=0,
)
helper.enable()

register_free_crossattn_upblock2d(
    diffusion_pipe,
    b1=1.3,
    b2=1.4,
    s1=0.9,
    s2=0.2,
)


@timeit
def generate_images(
    prompt: str,
    negatives: str = "",
    height: int = 1024,
    width: int = 1024,
    progress=gr.Progress(track_tqdm=True),
):
    TIME_LOG = {"GPU Power insert in W": POWER}

    if negatives is None:
        negatives = ""

    start_time = time.time()
    images = []
    with torch.inference_mode():
        image = diffusion_pipe(
            prompt=prompt,
            negative_prompt=negatives,
            num_inference_steps=INFER_STEPS,
            num_images_per_prompt=_images_per_prompt,
            height=height,
            width=width,
        ).images
        images.extend(image)

    consumed_time = time.time() - start_time
    TIME_LOG["Time in seconds"] = consumed_time
    TIME_LOG["Comsumed Watt hours"] = consumed_time * POWER / 3600
    TIME_LOG["Energy costs in cent"] = TIME_LOG["Comsumed Watt hours"] * 40 / 1000
    TIME_LOG["Device Name"] = GPU_NAME

    MD = json.dumps(TIME_LOG, indent=4)
    MD = "```json\n" + MD + "\n```"

    paths = []
    for x in range(len(images)):
        images[x].save(f"output_image_{x}.jpg", "JPEG", optimize=True)
        paths.append(f"output_image_{x}.jpg")

    return paths, MD


generate_images("cyborg style, golden retriever")
