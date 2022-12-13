from aaas.statics import MODEL_MAPPING
from aaas.audio_utils import get_model_and_processor

for mode in ["small", "medium", "large"]:
    for model in list(MODEL_MAPPING[mode].keys()):
        get_model_and_processor(model, mode)
