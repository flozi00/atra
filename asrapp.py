from atra.servers.speech_to_text import triton_server

triton_server.run()

from atra.gradio_utils.asr import build_asr_ui

build_asr_ui()
