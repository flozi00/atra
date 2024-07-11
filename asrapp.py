# adopted from amazing https://github.com/matatonic/openedai-whisper/tree/main
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import os
import torch
from transformers import pipeline
from typing import Optional, List
from fastapi import UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class OpenAIStub(FastAPI):
    def __init__(self) -> None:
        super().__init__()
        self.models = {}

        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


pipe = None
MODEL_DICT = {}
app = OpenAIStub()

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    dtype = torch.float32


def get_model(model_id) -> tuple[AutoModelForSpeechSeq2Seq, AutoProcessor]:
    global MODEL_DICT
    if model_id in MODEL_DICT:
        return MODEL_DICT[model_id]
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    model.to(device)

    if torch.cuda.is_available():
        import torch_tensorrt  # noqa
        model = torch.compile(
            model, mode="max-autotune", backend="torch_tensorrt", fullgraph=True
        )
    processor = AutoProcessor.from_pretrained(model_id)
    MODEL_DICT[model_id] = (model, processor)
    return model, processor


async def whisper(model_id, file, response_format: str, **kwargs):
    model, processor = get_model(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        chunk_length_s=30,
        torch_dtype=dtype,
    )

    result = pipe(await file.read(), batch_size=4, **kwargs)

    filename_noext, ext = os.path.splitext(file.filename)

    if response_format == "text":
        return PlainTextResponse(
            result["text"].strip(),
            headers={
                "Content-Disposition": f"attachment; filename={filename_noext}.txt"
            },
        )

    elif response_format == "json":
        return JSONResponse(
            content={"text": result["text"].strip()},
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename_noext}.json"
            },
        )

    elif response_format == "verbose_json":
        chunks = result["chunks"]

        response = {
            "task": kwargs["generate_kwargs"]["task"],
            # "language": "english",
            "duration": chunks[-1]["timestamp"][1],
            "text": result["text"].strip(),
        }
        if kwargs["return_timestamps"] == "word":
            response["words"] = [
                {
                    "word": chunk["text"].strip(),
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1],
                }
                for chunk in chunks
            ]
        else:
            response["segments"] = [
                {
                    "id": i,
                    # "seek": 0,
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1],
                    "text": chunk["text"].strip(),
                    # "tokens": [ ],
                    # "temperature": 0.0,
                    # "avg_logprob": -0.2860786020755768,
                    # "compression_ratio": 1.2363636493682861,
                    # "no_speech_prob": 0.00985979475080967
                }
                for i, chunk in enumerate(chunks)
            ]

        return JSONResponse(
            content=response,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"
            },
        )

    elif response_format == "srt":

        def srt_time(t):
            return "{:02d}:{:02d}:{:06.3f}".format(
                int(t // 3600), int(t // 60) % 60, t % 60
            ).replace(".", ",")

        return PlainTextResponse(
            "\n".join(
                [
                    f"{i}\n{srt_time(chunk['timestamp'][0])} --> {srt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                    for i, chunk in enumerate(result["chunks"], 1)
                ]
            ),
            media_type="text/srt; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={filename_noext}.srt"
            },
        )

    elif response_format == "vtt":

        def vtt_time(t):
            return "{:02d}:{:06.3f}".format(int(t // 60), t % 60)

        return PlainTextResponse(
            "\n".join(
                ["WEBVTT\n"]
                + [
                    f"{vtt_time(chunk['timestamp'][0])} --> {vtt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                    for chunk in result["chunks"]
                ]
            ),
            media_type="text/vtt; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={filename_noext}.vtt"
            },
        )


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile,
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),
    timestamp_granularities: List[str] = Form(["segment"]),
):
    global pipe

    kwargs = {"generate_kwargs": {"task": "transcribe"}}

    if language:
        kwargs["generate_kwargs"]["language"] = language.lower()
    # May work soon, https://github.com/huggingface/transformers/issues/27317
    #    if prompt:
    #        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs["generate_kwargs"]["temperature"] = temperature
    kwargs["generate_kwargs"]["do_sample"] = False
    kwargs["generate_kwargs"]["num_beams"] = 1

    if response_format == "verbose_json" and "word" in timestamp_granularities:
        kwargs["return_timestamps"] = "word"
    else:
        kwargs["return_timestamps"] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(model, file, response_format, **kwargs)


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile,
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),
):
    global pipe

    kwargs = {"generate_kwargs": {"task": "translate"}}

    # May work soon, https://github.com/huggingface/transformers/issues/27317
    #    if prompt:
    #        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs["generate_kwargs"]["temperature"] = temperature

    kwargs["generate_kwargs"]["do_sample"] = False
    kwargs["generate_kwargs"]["num_beams"] = 1

    kwargs["return_timestamps"] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(model, file, response_format, **kwargs)

@app.get("/v1")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7862)
