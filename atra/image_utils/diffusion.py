from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
)
from diffusers.models.cross_attention import AttnProcessor2_0

import torch
from atra.utils import timeit, ttl_cache
import GPUtil
import time
import io
from huggingface_hub import HfApi
import base64

api = HfApi()

pipe = None
refiner = None

TEMP_LIMIT = 65

import diffusers.pipelines.stable_diffusion_xl.watermark


def apply_watermark_dummy(self, images: torch.FloatTensor):
    return images


diffusers.pipelines.stable_diffusion_xl.watermark.StableDiffusionXLWatermarker.apply_watermark = (
    apply_watermark_dummy
)


def get_pipes():
    global pipe, refiner
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe.to("cuda")
    refiner.to("cuda")

    pipe.unet.set_attn_processor(AttnProcessor2_0())
    refiner.unet.set_attn_processor(AttnProcessor2_0())

    pipe.enable_xformers_memory_efficient_attention()
    refiner.enable_xformers_memory_efficient_attention()


@timeit
@ttl_cache(maxsize=128, ttl=60 * 60 * 6)
def generate_images(prompt: str, negatives: str = "", mode: str = "prototyping"):
    high_noise_frac = 0.7

    if pipe is None:
        get_pipes()

    if negatives is None:
        negatives = ""

    if mode == "prototyping":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        n_steps = 15
    else:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        n_steps = 60

    gpus = GPUtil.getGPUs()
    for gpu_num in range(len(gpus)):
        gpu = gpus[gpu_num]
        if gpu.temperature >= TEMP_LIMIT:
            faktor = int(gpu.temperature) - TEMP_LIMIT
            time.sleep(faktor * 10)  # wait for GPU to cool down

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

    if mode != "prototyping":
        buf = io.BytesIO()
        image.save(buf, format="png")
        byte_im = buf.getvalue()

        timestamp = str(int(time.time()))

        combined = prompt + "-->" + negatives + "-->" + timestamp

        encoded = base64.b64encode(combined.encode("utf-8")).decode("utf-8")

        api.upload_file(
            path_or_fileobj=byte_im,
            path_in_repo="images/{}.png".format(encoded),
            repo_id="flozi00/diffusions",
            repo_type="dataset",
            commit_message=prompt,
        )

    return image
