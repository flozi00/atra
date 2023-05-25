import gradio as gr

from atra.text_utils.chat import bot, user


def build_chatbot_ui():
    gr.Markdown("""## Open Assistant Chatbot
    This is a chatbot that can help you with your daily tasks. It is trained on a large amount of data and can answer a wide variety of questions. It is also able to learn from your conversations and improve over time.
    At the moment it is not possible to use external plugins and tools with this chatbot, but we are working on it.
    To use AI tools please select to tools tab above.""")
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
                submit = gr.Button("Submit")
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
        ],
        outputs=chatbot,
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
        ],
        outputs=chatbot,
        queue=True,
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)
