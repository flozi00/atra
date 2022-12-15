from evaluate import evaluator
from datasets import load_dataset
from transformers import pipeline
from aaas.statics import LANG_MAPPING
from aaas.audio_utils import get_model_and_processor

main_lang = "german"
model_config = "small"

if __name__ == "__main__":
    model, processor = get_model_and_processor(main_lang, model_config)
    try:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=LANG_MAPPING[main_lang], task="transcribe"
        )
    except Exception:
        pass

    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device_map="auto",
        torch_dtype="auto",
        num_beams=5,
        no_repeat_ngram_size=3,
    )

    task_evaluator = evaluator("automatic-speech-recognition")
    data = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "de",
        split="validation",
        cache_dir="data_cache",
        num_proc=1,
    )
    results = task_evaluator.compute(
        model_or_pipeline=transcriber,
        data=data,
        input_column="path",
        label_column="sentence",
        metric="wer",
    )
