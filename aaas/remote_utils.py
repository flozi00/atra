from aaas.backend_utils import CPU_BACKENDS, GPU_BACKENDS, inference_only
import random

if inference_only == False:
    from requests_futures.sessions import FuturesSession
    import numpy as np

    session = FuturesSession()

    def remote_inference(main_lang, model_config, data, premium=False):
        if len(GPU_BACKENDS) >= 1 and premium == True:
            target_port = random.choice(GPU_BACKENDS)
        elif len(CPU_BACKENDS) >= 1:
            target_port = random.choice(CPU_BACKENDS)
        else:
            target_port = 7860
        transcription = session.post(
            f"http://127.0.0.1:{target_port}/asr/{main_lang}/{model_config}/",
            data=np.asarray(data).tobytes(),
        )
        return transcription
