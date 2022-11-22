from aaas.audio_utils import (
    inference_asr,
    inference_denoise,
)
from aaas.backend_utils import (
    BACKENDS,
    get_used_ports,
    inference_only,
    master_node,
    master_pass,
    master_user,
)
from fastapi import FastAPI, Request
import uvicorn
import numpy as np
import os
import torch

app = FastAPI()


@app.post("/asr/{main_lang}/{model_config}/")
async def write(request: Request, main_lang, model_config):
    mydata = await request.body()
    audio = np.frombuffer(mydata, dtype=np.float32)

    audio = inference_denoise(audio)

    return inference_asr(
        data_batch=[audio], main_lang=main_lang, model_config=model_config
    )[0]


@app.get("/status/")
def get_status():
    return "success"


if inference_only == False:
    from aaas.gradio_utils import build_gradio

    @app.get("/get_free_port/")
    def get_set_port(password, device):
        global BACKENDS
        if password != master_pass:
            return "false password"

        merged_lists = get_used_ports()
        start_port = 7861
        while start_port in merged_lists:
            start_port += 1

        try:
            os.system(f"echo {master_pass}|sudo -S fuser -k {start_port}/tcp")
        except Exception as e:
            print(e)
        BACKENDS.append({"port": start_port, "device": device, "requests": 0})

        return start_port

    import gradio as gr

    app = gr.mount_gradio_app(app, build_gradio(), path="")
elif __name__ == "__main__":
    import requests

    forward_port = requests.get(
        f"https://{master_node}/get_free_port/",
        params={
            "password": master_pass,
            "device": "gpu" if torch.cuda.is_available() else "cpu",
        },
    ).json()
    forwarding_command = f'sshpass -p "{master_pass}" ssh -o StrictHostKeyChecking=no -fN -R {forward_port}:localhost:7860 {master_user}@{master_node}'
    print(forwarding_command)
    os.system(forwarding_command)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, log_level="info")
