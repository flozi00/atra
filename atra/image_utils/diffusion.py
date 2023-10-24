from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
import torch
from atra.utilities.stats import timeit
import time
import json
import diffusers.pipelines.stable_diffusion_xl.watermark
import os
from atra.image_utils.free_lunch_utils import (
    register_free_upblock2d,
    register_free_crossattn_upblock2d,
)


def apply_watermark_dummy(self, images: torch.FloatTensor):
    return images


diffusers.pipelines.stable_diffusion_xl.watermark.StableDiffusionXLWatermarker.apply_watermark = (
    apply_watermark_dummy
)

high_noise_frac = 0.7
INFER_STEPS = 80

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

GPU_ID = 0
POWER = 450
GPU_NAME = torch.cuda.get_device_name(GPU_ID)
if "H100" in GPU_NAME:
    POWER = 310
elif "A6000" in GPU_NAME:
    POWER = 300
elif "RTX 6000" in GPU_NAME:
    POWER = 240
elif "L40" in GPU_NAME:
    POWER = 350

PACKED_MODEL = os.getenv(
    "SDXL_FILE",
    "https://huggingface.co/jayparmr/DreamShaper_XL1_0_Alpha2/blob/main/dreamshaperXL10.safetensors",
)

diffusion_pipe = StableDiffusionXLPipeline.from_single_file(
    PACKED_MODEL,
    torch_dtype=torch.float16,
)

register_free_crossattn_upblock2d(
    diffusion_pipe,
    b1=1.3,
    b2=1.4,
    s1=0.9,
    s2=0.2,
)

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=diffusion_pipe.text_encoder_2,
    vae=diffusion_pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# set attention processor
refiner.unet.set_attn_processor(AttnProcessor2_0())
diffusion_pipe.unet.set_attn_processor(AttnProcessor2_0())

refiner.vae = torch.compile(refiner.vae, mode="reduce-overhead", fullgraph=True)

# change scheduler
diffusion_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    diffusion_pipe.scheduler.config
)
refiner.scheduler = EulerAncestralDiscreteScheduler.from_config(
    refiner.scheduler.config
)

# set to GPU
diffusion_pipe = diffusion_pipe.to(f"cuda:{GPU_ID}")
refiner.to(f"cuda:{GPU_ID}")


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

    consumed_time = time.time() - start_time
    TIME_LOG["Time in seconds"] = consumed_time
    TIME_LOG["Comsumed Watt hours"] = consumed_time * POWER / 3600
    TIME_LOG["Energy costs in cent"] = TIME_LOG["Comsumed Watt hours"] * 40 / 1000
    TIME_LOG["Device Name"] = GPU_NAME

    MD = json.dumps(TIME_LOG, indent=4)
    MD = "```json\n" + MD + "\n```"

    # image.save("output_image.jpg", "JPEG")
    # image = pathlib.Path("output_image.jpg")
    return image, MD
