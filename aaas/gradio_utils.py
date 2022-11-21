import gradio as gr
from aaas.audio_utils import run_transcription, LANG_MAPPING

langs = list(LANG_MAPPING.keys())


def build_gradio():
    ui = gr.Blocks()

    with ui:
        with gr.Tabs():
            with gr.TabItem("target language"):
                lang = gr.Radio(langs, value=langs[0])
            with gr.TabItem("model configuration"):
                model_config = gr.Radio(
                    choices=["monolingual", "multilingual"], value="monolingual"
                )

        with gr.Tabs():
            with gr.TabItem("Microphone"):
                mic = gr.Audio(source="microphone", type="filepath")
            with gr.TabItem("File"):
                audio_file = gr.Audio(source="upload", type="filepath")
            with gr.TabItem("URL"):
                video_url = gr.Textbox()

        with gr.Tabs():
            with gr.TabItem("Transcription"):
                transcription = gr.Textbox()
            with gr.TabItem("details"):
                chunks = gr.JSON()
            with gr.TabItem("Subtitled Video"):
                video = gr.Video()

        mic.change(
            fn=run_transcription,
            inputs=[mic, lang, model_config],
            outputs=[transcription, chunks, video],
            api_name="transcription",
        )
        audio_file.change(
            fn=run_transcription,
            inputs=[audio_file, lang, model_config],
            outputs=[transcription, chunks, video],
        )
        video_url.change(
            fn=run_transcription,
            inputs=[video_url, lang, model_config],
            outputs=[transcription, chunks, video],
        )

    return ui
