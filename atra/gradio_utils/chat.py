from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
from atra.text_utils.chat import bot, user
import gradio as gr


def build_chatbot_ui():
    ui = gr.Blocks(css=GLOBAL_CSS)
    with ui:
        GET_GLOBAL_HEADER()
        chatbot = gr.Chatbot().style(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                ).style(container=False)
            with gr.Column():
                with gr.Row():
                    ethernet = gr.Checkbox(label="Ethernet access", value=True)
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")

        submit_event = msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                ethernet,
            ],
            outputs=chatbot,
            queue=True,
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    ui.queue(concurrency_count=4, api_open=False)
    ui.launch(server_port=7860, **launch_args)
