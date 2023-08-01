from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0

import torch
from atra.utils import timeit, ttl_cache
import GPUtil
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

from transformers import pipeline

pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.unet.set_attn_processor(AttnProcessor2_0())
refiner.enable_xformers_memory_efficient_attention()
refiner = refiner.to("cuda")

MAPPINGS = {
    "general": {
        "class": StableDiffusionXLPipeline,
        "path": "https://huggingface.co/jayparmr/DreamShaper_XL1_0_Alpha2/blob/main/dreamshaperXL10.safetensors",
        "categories": ["general image"],
    },
    "mbbxl": {
        "class": StableDiffusionXLPipeline,
        "path": "https://huggingface.co/flozi00/diffusion-moe/blob/main/mbbxlUltimate_v10RC.safetensors",
        "categories": [
            "comic",
            "fantasy",
            "unreal",
            "mickey mouse",
            "superman",
            "marvel universum",
            "painting",
            "drawing",
        ],
    },
    "xl6": {
        "class": StableDiffusionXLPipeline,
        "path": "https://huggingface.co/flozi00/diffusion-moe/blob/main/xl6HEPHAISTOSSD10XLSFW_v10.safetensors",
        "categories": ["portrait", "animal", "human"],
    },
}

MERGED_CATEGORIES = []

for pipe_name in list(MAPPINGS.keys()):
    MERGED_CATEGORIES.extend(MAPPINGS[pipe_name]["categories"])


def get_pipes(expert):
    global MAPPINGS

    for pipe_name in list(MAPPINGS.keys()):
        if expert in MAPPINGS[pipe_name]["categories"]:
            if MAPPINGS[pipe_name].get("pipe", None) == None:
                MAPPINGS[pipe_name]["pipe"] = MAPPINGS[pipe_name][
                    "class"
                ].from_single_file(MAPPINGS[pipe_name]["path"], torch_dtype=torch.float16)
                MAPPINGS[pipe_name]["pipe"].enable_model_cpu_offload()
                MAPPINGS[pipe_name]["pipe"].unet.set_attn_processor(AttnProcessor2_0())
                MAPPINGS[pipe_name]["pipe"].enable_xformers_memory_efficient_attention()

            return MAPPINGS[pipe_name]["pipe"]


def query(payload):
    response = pipe(
        payload,
        candidate_labels=MERGED_CATEGORIES,
    )
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
    model_art = query(prompt)
    TIME_LOG["choosing expert"] = time.time() - start_time
    TIME_LOG["expert used"] = model_art
    TIME_LOG["watt-seconds"] = TIME_LOG["choosing expert"] * POWER

    start_time = time.time()
    pipe_to_use = get_pipes(model_art)
    TIME_LOG["model-loading"] = time.time() - start_time

    if negatives is None:
        negatives = ""

    start_time = time.time()
    if mode == "prototyping":
        pipe_to_use.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe_to_use.scheduler.config
        )
        n_steps = 15
    else:
        pipe_to_use.scheduler = EulerDiscreteScheduler.from_config(
            pipe_to_use.scheduler.config
        )
        n_steps = 60
    TIME_LOG["scheduler-loading"] = time.time() - start_time

    start_time = time.time()
    image = pipe_to_use(
        prompt=prompt,
        negative_prompt=negatives,
        num_inference_steps=n_steps,
    ).images[0]
    TIME_LOG["base-inference"] = time.time() - start_time
    TIME_LOG["watt-seconds"] += TIME_LOG["base-inference"] * POWER

    if mode != "prototyping":
        start_time = time.time()
        image = refiner(
            prompt=prompt,
            negative_prompt=negatives,
            num_inference_steps=int(n_steps / 3),
            image=image,
        ).images[0]
        TIME_LOG["refiner-inference"] = time.time() - start_time
        TIME_LOG["watt-seconds"] += TIME_LOG["refiner-inference"] * POWER

    TIME_LOG["watt-hours"] = TIME_LOG["watt-seconds"] / 3600

    TIME_LOG["energy-costs-euro"] = TIME_LOG["watt-hours"] / 1000 * 0.4
    TIME_LOG["energy-costs-cent"] = TIME_LOG["energy-costs-euro"] * 100

    return image, TIME_LOG
