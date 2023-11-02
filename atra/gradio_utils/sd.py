from atra.gradio_utils.ui import GLOBAL_CSS, GET_GLOBAL_HEADER, launch_args
import gradio as gr
from atra.image_utils.diffusion import generate_images

LORAS = [
    "",
    "KappaNeuro/isometric-cutaway",
    "jbilcke-hf/sdxl-cinematic-2",
    "TheLastBen/Papercut_SDXL",
    "minimaxir/sdxl-wrong-lora",
    "CiroN2022/cyber-ui",
    "goofyai/3d_render_style_xl",
    "ostris/ikea-instructions-lora-sdxl",
    "artificialguybr/LogoRedmond-LogoLoraForSDXL-V2",
    "goofyai/cyborg_style_xl",
]


def build_diffusion_ui() -> None:
    ui = gr.Blocks(css=GLOBAL_CSS)
    with ui:
        with gr.Row():
            GET_GLOBAL_HEADER()
        with gr.Column():
            images = gr.Image(height=512, width=512, label="Image")
            stats = gr.Markdown()

        with gr.Column():
            prompt = gr.Textbox(label="Prompt", info="Prompt of what you want to see")
            negatives = gr.Textbox(
                label="Negative Prompt",
                info="Prompt describing what you dont want to see, useful for refining image",
            )
            lora = gr.Dropdown(
                label="Lora",
                choices=LORAS,
                default="",
                info="Model to use for generating image",
            )

        prompt.submit(
            generate_images,
            inputs=[prompt, negatives, lora],
            outputs=[images, stats],
        )
        negatives.submit(
            generate_images,
            inputs=[prompt, negatives, lora],
            outputs=[images, stats],
        )

        gr.Examples(
            [
                [
                    "Isometric Cutaway - An image illustrating the installation guide for the cartridge. The background showcases a practical cartridge installation scenario, with illustrations or pictures demonstrating the correct installation steps. The accompanying text instructions are presented in a clear and understandable manner, assisting users in effortlessly completing the cartridge installation",
                    "KappaNeuro/isometric-cutaway",
                ],
                ["papercut, red fox", "TheLastBen/Papercut_SDXL"],
                ["3d style llama", "goofyai/3d_render_style_xl"],
                [
                    "LogoRedmAF, Icons for speech recognition app",
                    "artificialguybr/LogoRedmond-LogoLoraForSDXL-V2",
                ],
                ["cyborg style, golden retriever", "goofyai/cyborg_style_xl"],
                [
                    "High nation-geographic symmetrical close-up portrait shoot in green jungle of an expressive lizard, anamorphic lens, ultra-realistic, hyper-detailed, green-core, jungle-core",
                    "",
                ],
                ["Glowing jellyfish floating through a foggy forest at twilight", ""],
                [
                    "Elegant lavender garnish cocktail idea, cocktail glass, realistic, sharp focus, 8k high definition",
                    "",
                ],
                [
                    "General Artificial Intelligence in data center, futuristic concept art, 3d rendering",
                    "",
                ],
            ],
            inputs=[prompt, lora],
        )

    ui.queue(concurrency_count=1, api_open=False)
    ui.launch(**launch_args)
