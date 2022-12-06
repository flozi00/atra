import os
from aaas.gradio_utils import build_gradio

if __name__ == "__main__":
    ui = build_gradio()
    ui.queue()
    ui.launch(server_name="0.0.0.0", server_port=os.getenv("PORT", 7860))