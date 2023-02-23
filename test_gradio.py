from fastapi.testclient import TestClient
from app import app
import datasets
from transformers.pipelines.audio_utils import ffmpeg_read
from aaas.audio_utils.asr import inference_asr

client = TestClient(app)

ds = (
    datasets.load_dataset(
        "common_voice",
        "de",
        split="test",
        streaming=True,
    )
    ._resolve_features()
    .cast_column("audio", datasets.features.Audio(sampling_rate=16000, decode=False))
)

ds = iter(ds)
data = next(ds)


def test_get_client():
    response = client.get("/")
    assert response.status_code == 200


def test_get_gradio():
    response = client.get("/gradio/")
    assert response.status_code == 200


def test_asr():
    audio_data = ffmpeg_read(data["audio"]["bytes"], sampling_rate=16000)
    pred_str = inference_asr(audio_data, "german", "small")
    assert type(pred_str) == str
