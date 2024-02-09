from diffusers import (
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)

import torch
from atra.text_utils.prompts import IMAGES_ENHANCE_PROMPT
from atra.utilities.stats import timeit
import diffusers.pipelines.stable_diffusion_xl.watermark
from atra.image_utils.free_lunch_utils import (
    register_free_crossattn_upblock2d,
)
from DeepCache import DeepCacheSDHelper
import openai
import os

api_key = os.getenv("OAI_API_KEY", "")
base_url = os.getenv("OAI_BASE_URL", "https://api.together.xyz/v1")

if api_key != "":
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def apply_watermark_dummy(self, images: torch.FloatTensor):
    return images


diffusers.pipelines.stable_diffusion_xl.watermark.StableDiffusionXLWatermarker.apply_watermark = (
    apply_watermark_dummy
)

GPU_AVAILABLE = torch.cuda.is_available()

_images_per_prompt = 1
INFER_STEPS = 80
GPU_ID = 0


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
    height: int = 1024,
    width: int = 1024,
):
    if prompt.count(" ") < 15:
        try:
            chat_completion = client.chat.completions.create(
                model=os.getenv("OAI_MODEL", ""),
                messages=[
                    {"role": "system", "content": IMAGES_ENHANCE_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=64,
            )
            if isinstance(chat_completion, list):
                chat_completion = chat_completion[0]
            prompt = chat_completion.choices[0].message.content
        except:
            pass

    with torch.inference_mode():
        image = diffusion_pipe(
            prompt=prompt,
            num_inference_steps=INFER_STEPS,
            num_images_per_prompt=_images_per_prompt,
            height=height,
            width=width,
        ).images[0]

    return image, prompt
