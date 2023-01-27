from aaas.audio_utils.asr import inference_asr
from jiwer import cer


def similarity(ground, audio, main_lang, model_config):
    # run inference on the audio file
    result = inference_asr(audio, main_lang, model_config)

    # remove punctuation
    to_remove = [
        ".",
        ",",
        "!",
        "?",
        ";",
        ":",
        "-",
        "_",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "/",
        "\\",
        "|",
        "'",
        '"',
    ]

    for char in to_remove:
        ground = ground.replace(char, "")
        result = result.replace(char, "")

    # calculate distance between ground truth and result
    result_distance = cer(ground, result)

    return result_distance < 0.1
