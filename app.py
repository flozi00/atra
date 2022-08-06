import gradio as gr
from pyctcdecode import build_ctcdecoder
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, pipeline
import time



def run_transcription(audio, main_lang, hotword_categories):
    logs = ""
    start_time = time.time()
    transcription = ""
    chunks = []
    decoder = decoders[main_lang]

    p = pipeline("automatic-speech-recognition", model=model, tokenizer = tokenizer, feature_extractor = fextractor, decoder=decoder, chunk_length_s = 10, stride_length_s=(4, 1))

    logs += f"init vars time: {time.time() - start_time}\n"
    start_time = time.time()
    
    if audio is not None:
        hotwords = []
        for h in hotword_categories:
            with open(f"{h}.txt", "r") as f:
                words = f.read().splitlines()
                for w in words:
                    if len(w) >= 3:
                        hotwords.append(w.strip())

        logs += f"init hotwords time: {time.time() - start_time}\n"
        start_time = time.time()

        inferenced = p(audio, return_timestamps = "word", hotwords = hotwords, hotword_weight = 2)
        logs += f"inference time: {time.time() - start_time}\n"

        transcription = inferenced["text"]
        times = inferenced["chunks"]
        for i in range(len(times)):
            chunks.append({"text": times[i]["text"],"start": float(times[i]["timestamp"][0]), "end": float(times[i]["timestamp"][1])})

        return transcription, chunks, hotwords, logs
    else:
        return "", [], [], ""


"""
read the hotword categories from the index.txt file
"""


def get_categories():
    hotword_categories = []
    with open("index.txt", "r") as f:
        for line in f:
            if len(line) > 3:
                hotword_categories.append(line.strip())

    return hotword_categories


"""
Modell download and initialization
"""
modelid = "aware-ai/wav2vec2-xls-r-300m"
model = AutoModelForCTC.from_pretrained(modelid)
model.eval()

processor = Wav2Vec2Processor.from_pretrained(modelid)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(modelid)
fextractor = Wav2Vec2FeatureExtractor.from_pretrained(modelid)

"""
decoder stuff
"""
vocab_dict = processor.tokenizer.get_vocab()
sorted_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

decoders = {}
langs = ["german"]

for l in langs:
    decoder = build_ctcdecoder(
        list(sorted_dict.keys()),
        f"2glm-{l}.arpa",
    )
    decoders[l] = decoder


batch_size = 8


ui = gr.Blocks()

with ui:
    with gr.Tabs():
        with gr.TabItem("target language"):
            lang = gr.Radio(langs, value=langs[0])
        with gr.TabItem("hotword categories"):
            categories = gr.CheckboxGroup(choices=get_categories())

    with gr.Tabs():
        with gr.TabItem("Microphone"):
            mic = gr.Audio(source="microphone", type="filepath")
        with gr.TabItem("File"):
            audio_file = gr.Audio(source="upload", type="filepath")

    with gr.Tabs():
        with gr.TabItem("Transcription"):
            transcription = gr.Textbox()
        with gr.TabItem("Subtitle"):
            chunks = gr.JSON()
        with gr.TabItem("Hotwords"):
            hotwordlist = gr.JSON()
        with gr.TabItem("Logs"):
            logs = gr.Textbox()

    mic.change(
        fn=run_transcription,
        inputs=[mic, lang, categories],
        outputs=[transcription, chunks, hotwordlist, logs],
    )
    audio_file.change(
        fn=run_transcription,
        inputs=[audio_file, lang, categories],
        outputs=[transcription, chunks, hotwordlist, logs],
    )


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0")
