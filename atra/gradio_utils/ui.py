import os
import gradio as gr

GLOBAL_CSS = """
#hidden_stuff {display: none} 
"""


def GET_GLOBAL_HEADER():
    return gr.Markdown(
        "## This project is sponsored by [ ![PrimeLine](https://www.primeline-solutions.com/skin/frontend/default/theme566/images/primeline-solutions-logo.png) ](https://www.primeline-solutions.com/de/server/nach-einsatzzweck/gpu-rendering-hpc/)"
    )


auth_name = os.getenv("AUTH_NAME", None)
auth_password = os.getenv("AUTH_PASSWORD", None)

launch_args = dict(
    server_name="0.0.0.0",
    auth=(auth_name, auth_password)
    if auth_name is not None and auth_password is not None
    else None,
    show_api=False,
)
