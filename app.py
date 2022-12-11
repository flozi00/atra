import os
from aaas.gradio_utils import build_gradio

if __name__ == "__main__":
    ui = build_gradio()
    ui.queue(concurrency_count=2)
    ui.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), max_threads=16)