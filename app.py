import os

mode = os.getenv("SERVERMODE", "APP")

if mode == "APP":
    if __name__ == "__main__":
        from aaas.gradio_utils import build_gradio

        ui = build_gradio()
        ui.queue(concurrency_count=20)
        ui.launch(
            server_name="0.0.0.0",
            server_port=int(os.getenv("PORT", 7860)),
            max_threads=32,
        )
else:
    print("Worker Mode")
    import base64

    from transformers.pipelines.audio_utils import ffmpeg_read

    from aaas.audio_utils import inference_asr
    from aaas.datastore import get_audio_queue, set_in_progress, set_transkript

    while True:
        task = get_audio_queue()
        if task is not False:
            set_in_progress(task.hs)
            audio = base64.b64decode(task.data.encode("UTF-8"))
            audio = ffmpeg_read(audio, 16000)
            result = inference_asr(
                data_batch=[audio],
                main_lang=task.main_lang,
                model_config=task.model_config,
            )[0]
            set_transkript(task.hs, result)
