from optimum.pipelines import pipeline
from aaas.audio_utils import LANG_MAPPING

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

trans_pipes = {
    "Helsinki-NLP/opus-mt-en-de": pipeline(
        "translation", model=f"Helsinki-NLP/opus-mt-en-de"
    ),
    "Helsinki-NLP/opus-mt-de-en": pipeline(
        "translation", model=f"Helsinki-NLP/opus-mt-de-en"
    ),
}


def translate(text, source, target):
    global trans_pipes

    if source == target:
        return text

    model_id = f"Helsinki-NLP/opus-mt-{source}-{target}"
    trans_pipe = trans_pipes.get(model_id, None)
    if trans_pipe is None:
        trans_pipe = pipeline(
            "translation",
            model=model_id,
        )
        trans_pipes[model_id] = trans_pipe

    translated = trans_pipe(text)[0]["translation_text"]

    return translated


def summarize(main_lang, text):
    en_version = translate(text, LANG_MAPPING[main_lang], "en")

    summarization = summarizer(
        en_version, max_length=130, min_length=30, do_sample=False
    )[0]["summary_text"]

    summarization = translate(summarization, "en", LANG_MAPPING[main_lang])

    return summarization
