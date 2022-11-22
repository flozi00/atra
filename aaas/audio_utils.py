import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import os
import torch.quantization
import torch.nn as nn
from optimum.bettertransformer import BetterTransformer
from aaas.backend_utils import inference_only, check_nodes
from aaas.statics import *

if inference_only == False:
    from aaas.video_utils import merge_subtitles
    from aaas.text_utils import translate
    from aaas.remote_utils import remote_inference
    import subprocess
    from aaas.silero_vad import silero_vad

    model_vad, get_speech_timestamps = silero_vad(True)


def inference_denoise(audio):
    return audio


def inference_asr(data_batch, main_lang: str, model_config: str) -> str:
    transcription = []
    if model_config == "multilingual":
        model, processor = get_model_and_processor("universal")
    else:
        model, processor = get_model_and_processor(main_lang)

    for data in data_batch:
        input_values = processor.feature_extractor(
            data,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=True,
        ).input_features

        with torch.inference_mode():
            if torch.cuda.is_available():
                input_values = input_values.to("cuda").half()
            predicted_ids = model.generate(
                input_values,
                max_length=int(((len(data) / 16000) * 12) / 2) + 10,
                use_cache=True,
                no_repeat_ngram_size=1,
                num_beams=20,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language=LANG_MAPPING[main_lang], task="transcribe"
                ),
            )

        transcription.append(
            processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        )

    return transcription


def get_model_and_processor(lang: str):
    try:
        model_id = MODEL_MAPPING[lang]["name"]
    except:
        lang = "universal"
        model_id = MODEL_MAPPING[lang]["name"]

    model = MODEL_MAPPING[lang].get("model", None)
    processor = MODEL_MAPPING[lang].get("processor", None)

    if processor == None:
        processor = WhisperProcessor.from_pretrained(model_id)
        MODEL_MAPPING[lang]["processor"] = processor

    if model == None:
        model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
        if torch.cuda.is_available():
            model = model.to("cuda").half()
        model = BetterTransformer.transform(model)

        if torch.cuda.is_available() == False:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )

        MODEL_MAPPING[lang]["model"] = model

    return model, processor


# copied from https://github.com/huggingface/transformers
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
        with subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        ) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError(
            "ffmpeg was not found but is required to load audio files from filename"
        ) from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


def batch_audio_by_silence(audio_batch):
    new_batch = []
    tmp_audio = []
    for b in audio_batch:
        if len(tmp_audio) + len(b) < 30 * 16000:
            tmp_audio.extend(b)
        elif len(b) > 28 * 16000:
            new_batch.append(tmp_audio)
            tmp_audio = []
            new_batch.append(b)
        else:
            new_batch.append(tmp_audio)
            tmp_audio = []

    if tmp_audio != []:
        new_batch.append(tmp_audio)

    return new_batch


def run_transcription(audio, main_lang, model_config, target_lang=""):
    chunks = []
    file = None
    full_transcription = {"target_text": ""}
    if target_lang == "":
        target_lang = main_lang
    check_nodes()

    if audio is not None and len(audio) > 3:
        if isinstance(audio, str):
            audio_name = audio.split(".")[-2]
            audio_path = audio
            extension = audio_path.split(".")[-1]

            if extension in ["mp4"]:
                do_stream = True
            else:
                do_stream = False

            with open(audio, "rb") as f:
                payload = f.read()

            audio = ffmpeg_read(payload, sampling_rate=16000)

        speech_timestamps = get_speech_timestamps(
            audio,
            model_vad,
            sampling_rate=16000,
            min_silence_duration_ms=250,
            speech_pad_ms=200,
        )
        audio_batch = [
            audio[speech_timestamps[st]["start"] : speech_timestamps[st]["end"]]
            for st in range(len(speech_timestamps))
        ]

        if do_stream == False:
            audio_batch = batch_audio_by_silence(audio_batch)

        transcription = []
        for data in audio_batch:
            transcription.append(
                remote_inference(
                    main_lang=main_lang,
                    model_config=model_config,
                    data=data,
                    premium= not do_stream,
                )
            )

        for x in range(len(audio_batch)):
            response = transcription[x].result()
            chunks.append(
                {
                    "native_text": response.json(),
                    "start_timestamp": (speech_timestamps[x]["start"] / 16000) - 0.1,
                    "stop_timestamp": (speech_timestamps[x]["end"] / 16000) - 0.5,
                }
            )

        chunks = sorted(chunks, key=lambda d: d["start_timestamp"])

        for c in range(len(chunks)):
            chunks[c]["target_text"] = translate(
                chunks[c]["native_text"],
                LANG_MAPPING[main_lang],
                LANG_MAPPING[target_lang],
            )
            full_transcription["target_text"] += chunks[c]["target_text"] + "\n"

        if do_stream == True:
            file = merge_subtitles(chunks, audio_path, audio_name)

        os.remove(audio_path)

    return full_transcription["target_text"], chunks, file
