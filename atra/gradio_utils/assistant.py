import gradio as gr

from atra.text_utils.chat import bot, user


def build_chatbot_ui():
    gr.Markdown(
        """## Open Assistant Chatbot
    This is an open assistant chatbot. It is designed to be a general purpose chatbot that can be used for any task.<br>
    To use AI for non textual tasks like speech recognition please select to tools tab above.<br>"""
    )
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
                ethernet = gr.Checkbox(label="Ethernet access")
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
