from aaas.audio_utils.asr import inference_asr
from jiwer import cer
import re


def similarity(ground, audio, main_lang, model_config):
    # run inference on the audio file
    result = inference_asr(audio, main_lang, model_config)

    ground = re.sub(r"[^\w\s]", "", ground)
    result = re.sub(r"[^\w\s]", "", result)

    # calculate distance between ground truth and result
    result_distance = cer(ground, result)

    return result_distance < 0.2
