from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)
from diffusers.models.cross_attention import AttnProcessor2_0

import torch
from atra.utils import timeit, ttl_cache
import GPUtil
import time
import subprocess
import json

general_pipe = None
exterior_pipe = None
interior_pipe = None
refiner = None

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

import diffusers.pipelines.stable_diffusion_xl.watermark


def apply_watermark_dummy(self, images: torch.FloatTensor):
    return images


diffusers.pipelines.stable_diffusion_xl.watermark.StableDiffusionXLWatermarker.apply_watermark = (
    apply_watermark_dummy
)

subprocess = subprocess.Popen("gpustat -P --json", shell=True, stdout=subprocess.PIPE)
subprocess_return = subprocess.stdout.read()

POWER = json.loads(subprocess_return)["gpus"][0]["enforced.power.limit"]

def get_pipes():
    global general_pipe, refiner, exterior_pipe, interior_pipe

    general_pipe = StableDiffusionXLPipeline.from_single_file("https://huggingface.co/jayparmr/DreamShaper_XL1_0_Alpha2/blob/main/dreamshaperXL10.safetensors", torch_dtype=torch.float16, use_safetensors=True,)
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=general_pipe.text_encoder_2,
        vae=general_pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    interior_pipe = StableDiffusionPipeline.from_single_file("https://huggingface.co/flozi00/diffusion-moe/blob/main/xsarchitectural_v11.ckpt")
    exterior_pipe = StableDiffusionPipeline.from_single_file("https://huggingface.co/flozi00/diffusion-moe/blob/main/xsarchitecturalv3com_v31InSafetensor.safetensors")


    general_pipe.enable_model_cpu_offload()
    refiner.enable_model_cpu_offload()
    exterior_pipe.enable_model_cpu_offload()
    interior_pipe.enable_model_cpu_offload()


    general_pipe.unet.set_attn_processor(AttnProcessor2_0())
    refiner.unet.set_attn_processor(AttnProcessor2_0())
    interior_pipe.unet.set_attn_processor(AttnProcessor2_0())
    exterior_pipe.unet.set_attn_processor(AttnProcessor2_0())

    general_pipe.enable_xformers_memory_efficient_attention()
    refiner.enable_xformers_memory_efficient_attention()
    interior_pipe.enable_xformers_memory_efficient_attention()
    exterior_pipe.enable_xformers_memory_efficient_attention()


from transformers import pipeline

pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
def query(payload):
    response = pipe(payload, candidate_labels= ["interior design", "exterior design", "general image", "portrait"])
    return response["labels"][0]


@timeit
@ttl_cache(maxsize=128, ttl=60 * 60 * 6)
def generate_images(prompt: str, negatives: str = "", mode: str = "prototyping"):
    TIME_LOG = {"gpu-power": POWER}

    if mode != "prototyping":
        for pattern in BAD_PATTERNS:
            if pattern in prompt:
                raise "NSFW prompt not allowed"
            
    gpus = GPUtil.getGPUs()
    for gpu_num in range(len(gpus)):
        gpu = gpus[gpu_num]
        if gpu.temperature >= TEMP_LIMIT:
            faktor = int(gpu.temperature) - TEMP_LIMIT
            time.sleep(faktor * 10)  # wait for GPU to cool down

    start_time = time.time()
    if general_pipe is None:
        get_pipes()
    TIME_LOG["model-loading"] = time.time() - start_time

    if negatives is None:
        negatives = ""

    start_time = time.time()
    if mode == "prototyping":
        general_pipe.scheduler = DPMSolverSinglestepScheduler.from_config(general_pipe.scheduler.config)
        interior_pipe.scheduler = DPMSolverSinglestepScheduler.from_config(interior_pipe.scheduler.config)
        exterior_pipe.scheduler = DPMSolverSinglestepScheduler.from_config(exterior_pipe.scheduler.config)
        n_steps = 15
    else:
        general_pipe.scheduler = EulerDiscreteScheduler.from_config(general_pipe.scheduler.config)
        interior_pipe.scheduler = EulerDiscreteScheduler.from_config(interior_pipe.scheduler.config)
        exterior_pipe.scheduler = EulerDiscreteScheduler.from_config(exterior_pipe.scheduler.config)
        n_steps = 60
    TIME_LOG["scheduler-loading"] = time.time() - start_time

    start_time = time.time()
    model_art = query(prompt)
    TIME_LOG["choosing expert"] = time.time() - start_time
    TIME_LOG["expert used"] = model_art
    TIME_LOG["watt-seconds"] = TIME_LOG["choosing expert"] * POWER

    if model_art == "interior design":
        pipe_to_use = interior_pipe
    elif model_art == "exterior design":
        pipe_to_use = exterior_pipe
    else:
        pipe_to_use = general_pipe

    start_time = time.time()
    image = pipe_to_use(
        prompt=prompt,
        negative_prompt = negatives,
        num_inference_steps=n_steps,
    ).images[0]
    TIME_LOG["base-inference"] = time.time() - start_time
    TIME_LOG["watt-seconds"] += TIME_LOG["base-inference"] * POWER
    
    if mode != "prototyping":
        start_time = time.time()
        image = refiner(
            prompt=prompt,
            negative_prompt = negatives,
            num_inference_steps=int(n_steps/3),
            image=image,
        ).images[0]
        TIME_LOG["refiner-inference"] = time.time() - start_time
        TIME_LOG["watt-seconds"] += TIME_LOG["refiner-inference"] * POWER

    TIME_LOG["watt-hours"] = TIME_LOG["watt-seconds"]/3600

    TIME_LOG["energy-costs-euro"] = TIME_LOG["watt-hours"]/1000*0.4
    TIME_LOG["energy-costs-cent"] = TIME_LOG["energy-costs-euro"]*100

    return image, TIME_LOG
