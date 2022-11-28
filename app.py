from fastapi import FastAPI
import uvicorn
import os
from aaas.gradio_utils import build_gradio
import gradio as gr

app = FastAPI()


@app.get("/status/")
def get_status():
    return "success"


app = gr.mount_gradio_app(app, build_gradio(), path="")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 7860), log_level="info")
