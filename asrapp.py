#adopted from amazing https://github.com/matatonic/openedai-whisper/tree/main
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import os
import sys
import argparse

import torch
from transformers import pipeline
from typing import Optional, List
from fastapi import UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn
import torch_tensorrt # noqa
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
            allow_headers=["*"]
        )

        @self.get('/v1/billing/usage')
        @self.get('/v1/dashboard/billing/usage')
        async def handle_billing_usage():
            return { 'total_usage': 0 }

        @self.get("/", response_class=PlainTextResponse)
        @self.head("/", response_class=PlainTextResponse)
        @self.options("/", response_class=PlainTextResponse)
        async def root():
            return PlainTextResponse(content="", status_code=200 if self.models else 503)

        @self.get("/health")
        async def health():
            return {"status": "ok" if self.models else "unk" }

        @self.get("/v1/models")
        async def get_model_list():
            return self.model_list()

        @self.get("/v1/models/{model}")
        async def get_model_info(model_id: str):
            return self.model_info(model_id)

    def register_model(self, name: str, model: str = None) -> None:
        self.models[name] = model if model else name

    def deregister_model(self, name: str) -> None:
        if name in self.models:
            del self.models[name]

    def model_info(self, model: str) -> dict:
        result = {
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": "user"
        }
        return result

    def model_list(self) -> dict:
        if not self.models:
            return {}
        
        result = {
            "object": "list",
            "data": [ self.model_info(model) for model in list(set(self.models.keys() | self.models.values())) if model ]
        }

        return result
    

pipe = None
app = OpenAIStub()

async def whisper(file, response_format: str, **kwargs):
    global pipe

    result = pipe(await file.read(), batch_size=16, **kwargs)

    filename_noext, ext = os.path.splitext(file.filename)

    if response_format == "text":
        return PlainTextResponse(result["text"].strip(), headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})

    elif response_format == "json":
        return JSONResponse(content={ 'text': result['text'].strip() }, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})
    
    elif response_format == "verbose_json":
        chunks = result["chunks"]

        response = {
            "task": kwargs['generate_kwargs']['task'],
            #"language": "english",
            "duration": chunks[-1]['timestamp'][1],
            "text": result["text"].strip(),
        }
        if kwargs['return_timestamps'] == 'word':
            response['words'] = [{'word': chunk['text'].strip(), 'start': chunk['timestamp'][0], 'end': chunk['timestamp'][1] } for chunk in chunks ]
        else:
            response['segments'] = [{
                    "id": i,
                    #"seek": 0,
                    'start': chunk['timestamp'][0],
                    'end': chunk['timestamp'][1],
                    'text': chunk['text'].strip(),
                    #"tokens": [ ],
                    #"temperature": 0.0,
                    #"avg_logprob": -0.2860786020755768,
                    #"compression_ratio": 1.2363636493682861,
                    #"no_speech_prob": 0.00985979475080967
            } for i, chunk in enumerate(chunks) ]
        
        return JSONResponse(content=response, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"})

    elif response_format == "srt":
            def srt_time(t):
                return "{:02d}:{:02d}:{:06.3f}".format(int(t//3600), int(t//60)%60, t%60).replace(".", ",")

            return PlainTextResponse("\n".join([ f"{i}\n{srt_time(chunk['timestamp'][0])} --> {srt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for i, chunk in enumerate(result["chunks"], 1) ]), media_type="text/srt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.srt"})

    elif response_format == "vtt":
            def vtt_time(t):
                return "{:02d}:{:06.3f}".format(int(t//60), t%60)
            
            return PlainTextResponse("\n".join(["WEBVTT\n"] + [ f"{vtt_time(chunk['timestamp'][0])} --> {vtt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for chunk in result["chunks"] ]), media_type="text/vtt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.vtt"})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
        file: UploadFile,
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
        timestamp_granularities: List[str] = Form(["segment"])
    ):
    global pipe

    kwargs = {'generate_kwargs': {'task': 'transcribe'}}

    if language:
        kwargs['generate_kwargs']["language"] = language
# May work soon, https://github.com/huggingface/transformers/issues/27317
#    if prompt:
#        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs['generate_kwargs']["temperature"] = temperature
        kwargs['generate_kwargs']['do_sample'] = True

    if response_format == "verbose_json" and 'word' in timestamp_granularities:
        kwargs['return_timestamps'] = 'word'
    else:
        kwargs['return_timestamps'] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(file, response_format, **kwargs)


@app.post("/v1/audio/translations")
async def translations(
        file: UploadFile,
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
    ):
    global pipe

    kwargs = {'generate_kwargs': {"task": "translate"}}

# May work soon, https://github.com/huggingface/transformers/issues/27317
#    if prompt:
#        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs['generate_kwargs']["temperature"] = temperature
        kwargs['generate_kwargs']['do_sample'] = True

    kwargs['return_timestamps'] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(file, response_format, **kwargs)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    
    MODEL_ID = os.getenv("ASR_MODEL", "primeline/whisper-large-v3-german")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
    )
    model.to(device)

    model = torch.compile(
        model, mode="max-autotune", backend="torch_tensorrt", fullgraph=True
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    pipe = pipeline("automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor, 
    device=device, chunk_length_s=30, torch_dtype=dtype)

    app.register_model('whisper-1', MODEL_ID)

    uvicorn.run(app, host="0.0.0.0", port=7862)
