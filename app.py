import gradio as gr
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
import subprocess
import numpy as np

def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio

def run_transcription(audio, main_lang, hotword_categories):
    transcription = ""
    chunks = []
    decoder = decoders[main_lang]
    if(audio is not None):
        hotwords = []
        for h in hotword_categories:
            with open(f"{h}.txt", "r") as f:
                words = f.read().splitlines()
                for w in words:
                    if(len(w) >= 3):
                        hotwords.append(w.strip())

        with open(audio, "rb") as f:
            payload = f.read()
        audio = ffmpeg_read(payload, sampling_rate=16000)

        speech_timestamps = get_speech_timestamps(audio, model_vad, sampling_rate=16000)
        audio_batch = [audio[speech_timestamps[st]["start"]:speech_timestamps[st]["end"]] for st in range(len(speech_timestamps))]

        for x in range(0, len(audio_batch), batch_size):
            data = audio_batch[x:x+batch_size]
            
            input_values = processor(data, sampling_rate=16000, return_tensors="pt", padding=True).input_values
            with torch.inference_mode():
                logits = model(input_values).logits


            for y in range(len(logits)):
                beams = decoder.decode_beams(logits.cpu().numpy()[y], beam_width=250, hotword_weight=150, hotwords=hotwords)

                offset = speech_timestamps[x+y]["start"]/processor.feature_extractor.sampling_rate
                top_beam = beams[0]
                transcription_beam, lm_state, indices, logit_score, lm_score = top_beam
                transcription_beam = transcription_beam.replace("\"", "")
                transcription += transcription_beam + " "
                word_offsets = []
                chunk_offset = indices
                for word, (start_offset, end_offset) in chunk_offset:
                    word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})

                for item in word_offsets:
                    start = item["start_offset"] * 320
                    start /= processor.feature_extractor.sampling_rate

                    stop = item["end_offset"] * 320
                    stop /= processor.feature_extractor.sampling_rate

                    chunks.append({"text": item["word"], "timestamp": (start+offset, stop+offset)})

        return transcription, chunks, hotwords
    else:
        return "", [], []


"""
read the hotword categories from the index.txt file
"""
def get_categories():
    hotword_categories = []
    with open("index.txt", "r") as f:
        for line in f:
            if(len(line) > 3):
                hotword_categories.append(line.strip())

    return hotword_categories


"""
VAD download and initialization
"""
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False, onnx=True)

(get_speech_timestamps,
 _, read_audio,
 *_) = utils


"""
Modell download and initialization
"""
model = AutoModelForCTC.from_pretrained("aware-ai/wav2vec2-xls-r-300m")
model.eval()

processor = Wav2Vec2Processor.from_pretrained("aware-ai/wav2vec2-xls-r-300m")

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


    mic.change(fn=run_transcription, inputs=[mic, lang, categories], outputs=[transcription, chunks, hotwordlist])
    audio_file.change(fn=run_transcription, inputs=[audio_file, lang, categories], outputs=[transcription, chunks, hotwordlist])

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0")