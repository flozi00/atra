from pytriton.decorators import batch, first_value, group_by_values
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
import base64
import io
import numpy as np

from atra.audio_utils.asr import speech_recognition


@batch
@group_by_values("language")
def _infer_fn(
    language: np.ndarray,
    audio: np.ndarray,
):
    languages = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in language]
    audios = [base64.b64decode(p[0]) for p in audio]

    outputs = []

    for i in range(len(languages)):
        transcription = speech_recognition(data=audios[i], language=languages[i])
        outputs.append(np.char.encode(np.array([transcription]), "utf-8"))

    return {"transcription": np.array(outputs)}


config = TritonConfig(exit_on_error=True)

triton_server = Triton(config=config)
triton_server.bind(
    model_name="Whisper",
    infer_func=_infer_fn,
    inputs=[
        Tensor(name="language", dtype=np.bytes_, shape=(1,)),
        Tensor(name="audio", dtype=np.bytes_, shape=(1,)),
    ],
    outputs=[
        Tensor(name="transcription", dtype=np.bytes_, shape=(1,)),
    ],
    config=ModelConfig(
        max_batch_size=4,
        batcher=DynamicBatcher(
            max_queue_delay_microseconds=100,
        ),
    ),
    strict=True,
)
