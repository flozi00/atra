from aaas.statics import MODEL_MAPPING
from aaas.audio_utils import get_model_and_processor

for model in list(MODEL_MAPPING.keys()):
    if(model != "universal"):
        get_model_and_processor(model)