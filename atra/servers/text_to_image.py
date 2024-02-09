from pytriton.decorators import batch, first_value, group_by_values
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
import base64
import io
from atra.image_utils.diffusion import generate_images
import numpy as np

IMAGE_FORMAT = "JPEG"


def _encode_image_to_base64(image):
    raw_bytes = io.BytesIO()
    image.save(raw_bytes, IMAGE_FORMAT)
    raw_bytes.seek(0)  # return to the start of the buffer
    return base64.b64encode(raw_bytes.read())


@batch
@group_by_values("img_size")
@first_value("img_size")
def _infer_fn(
    prompt: np.ndarray,
    img_size: np.int64,
):
    prompts = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in prompt]

    outputs = []
    used_prompts = []
    for _prompt in prompts:
        image, prompt = generate_images(prompt=_prompt, height=img_size, width=img_size)
        raw_data = _encode_image_to_base64(image)
        used_prompts.append(np.char.encode(np.array([prompt]), "utf-8"))
        outputs.append([raw_data])

    return {"image": np.array(outputs), "prompt": np.array(used_prompts)}


config = TritonConfig(exit_on_error=True)

triton_server = Triton(config=config)
triton_server.bind(
    model_name="SDXL",
    infer_func=_infer_fn,
    inputs=[
        Tensor(name="prompt", dtype=np.bytes_, shape=(1,)),
        Tensor(name="img_size", dtype=np.int64, shape=(1,)),
    ],
    outputs=[
        Tensor(name="image", dtype=np.bytes_, shape=(1,)),
        Tensor(name="prompt", dtype=np.bytes_, shape=(1,)),
    ],
    config=ModelConfig(
        max_batch_size=4,
        batcher=DynamicBatcher(
            max_queue_delay_microseconds=100,
        ),
    ),
    strict=True,
)
