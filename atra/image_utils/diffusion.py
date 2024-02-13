from diffusers import (
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL,
)

import torch
from atra.text_utils.prompts import IMAGES_ENHANCE_PROMPT
from atra.utilities.stats import timeit
import diffusers.pipelines.stable_diffusion_xl.watermark
from DeepCache import DeepCacheSDHelper
import openai
import os
import model_navigator as nav
from transformers.modeling_outputs import BaseModelOutputWithPooling
import transformers, diffusers

DEVICE = torch.device("cuda")

GPU_AVAILABLE = torch.cuda.is_available()

_images_per_prompt = 1
INFER_STEPS = 120
GPU_ID = 0

MODEL_ID = "dataautogpt3/ProteusV0.3"

# workaround to make transformers use the same device as model navigator
transformers.modeling_utils.get_parameter_device = lambda parameter: DEVICE
diffusers.models.modeling_utils.get_parameter_device = lambda parameter: DEVICE

nav.inplace_config.mode = "optimize"
nav.inplace_config.min_num_samples = INFER_STEPS * 2

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

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)


pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    vae=vae,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to(DEVICE)

optimize_config = nav.OptimizeConfig(
    batching=False,
    target_formats=(nav.Format.TENSORRT,),
    runners=(
        # "TorchCUDA",
        "TensorRT",
    ),
    custom_configs=[
        nav.TensorRTConfig(precision=nav.TensorRTPrecision.FP16, optimization_level=5)
    ],
)


# For outputs that are not primitive types (float, int, bool, str) or tensors and list, dict, tuples combinations of those.
# we need to provide a mapping to a desired output type. CLIP output is BaseModelOutputWithPooling, which inherits from dict.
# Model Navigator will recognize that the return type is a dict and will return it, but we need to provide a mapping to BaseModelOutputWithPooling.
def clip_output_mapping(output):
    return BaseModelOutputWithPooling(**output)


pipe.text_encoder = nav.Module(
    pipe.text_encoder,
    optimize_config=optimize_config,
    output_mapping=clip_output_mapping,
)
pipe.unet = nav.Module(
    pipe.unet,
    optimize_config=optimize_config,
)
pipe.vae.decoder = nav.Module(
    pipe.vae.decoder,
    optimize_config=optimize_config,
)

pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)

helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(
    cache_interval=3,
    cache_branch_id=0,
)
helper.enable()


@timeit
def generate_images(
    prompt: str,
    height: int = 1024,
    width: int = 1024,
):
    negative_prompt = "bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"

    if prompt.count(" ") < 15:
        try:
            if api_key != "":
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

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=INFER_STEPS,
        num_images_per_prompt=_images_per_prompt,
        height=height,
        width=width,
        guidance_scale=7,
    ).images[0]

    return image, prompt
