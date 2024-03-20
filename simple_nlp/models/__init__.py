from transformers import AutoConfig
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Wav2Vec2BertForCTC,
)


def get_model(
    model_name: str,
    processor_name: str = None,
    vocab_size=None,
    processor=None,
):
    kwargs = {}
    use_flash_v2 = True
    # get the config of the base model and extract the model type from it
    conf = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )

    ctc_model = False
    keys = ["wav2vec", "w2v"]
    for key in keys:
        if key in model_name:
            ctc_model = True
            conf.attention_dropout=0.0
            conf.hidden_dropout=0.0
            conf.feat_proj_dropout=0.0
            conf.mask_time_prob=0.0
            conf.layerdrop=0.0
            conf.ctc_loss_reduction="mean"
            conf.add_adapter = True
            use_flash_v2 = False
            break
    model_class = AutoModelForSpeechSeq2Seq if ctc_model is False else Wav2Vec2BertForCTC
    tok_class = AutoProcessor


    if processor is None:
        processor = tok_class.from_pretrained(
            model_name if processor_name is None else processor_name,
            legacy=False,
            trust_remote_code=True,
        )

    if use_flash_v2:
        kwargs["attn_implementation"] = "sdpa"
    #else:
    #    kwargs["attn_implementation"] = "sdpa"
    
    if vocab_size is not None:
        conf.vocab_size = vocab_size
        kwargs["ignore_mismatched_sizes"]=True

    # load the pre-trained model and check if its 8-bit compatible
    model = model_class.from_pretrained(
        model_name,
        config=conf,
        **kwargs,
    )

    try:
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
            print("Setting pad token to eos token")
    except Exception:
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            print("Setting pad token to eos token")


    model = model.train()

    return model, processor
