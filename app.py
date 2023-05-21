import os

from atra.gradio_utils import build_gradio

ui = build_gradio()


if __name__ == "__main__":
    auth_name = os.getenv("AUTH_NAME", None)
    auth_password = os.getenv("AUTH_PASSWORD", None)

    ui.launch(
        enable_queue=True,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        auth=(auth_name, auth_password)
        if auth_name is not None and auth_password is not None
        else None,
    )
