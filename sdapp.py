from atra.servers.text_to_image import triton_server

triton_server.run()

from atra.gradio_utils.sd import build_diffusion_ui

build_diffusion_ui()
