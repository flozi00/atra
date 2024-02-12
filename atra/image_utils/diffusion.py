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
import model_navigator as nav
from transformers.modeling_outputs import BaseModelOutputWithPooling
import transformers, diffusers

DEVICE = torch.device("cuda")

# workaround to make transformers use the same device as model navigator
transformers.modeling_utils.get_parameter_device = lambda parameter: DEVICE
diffusers.models.modeling_utils.get_parameter_device = lambda parameter: DEVICE

nav.inplace_config.mode = "optimize"
nav.inplace_config.min_num_samples = 100

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
INFER_STEPS = 20
GPU_ID = 0


pipe = StableDiffusionXLPipeline.from_pretrained(
    "dataautogpt3/ProteusV0.2",
    torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
)

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


pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

pipe.fuse_qkv_projections()

# change scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(
    cache_interval=int(INFER_STEPS / 10),
    cache_branch_id=0,
)
helper.enable()

register_free_crossattn_upblock2d(
    pipe,
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
        num_inference_steps=INFER_STEPS,
        num_images_per_prompt=_images_per_prompt,
        height=height,
        width=width,
    ).images[0]

    return image, prompt
