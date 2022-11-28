from aaas.backend_utils import inference_only, get_best_node, increase_queue, fallback_url

if inference_only == False:
    from requests_futures.sessions import FuturesSession
    import numpy as np

    session = FuturesSession()

    def remote_inference(main_lang, model_config, data, premium=False):
        target_port = get_best_node(premium=premium)
        if target_port != False:
            url = f"http://127.0.0.1:{target_port}/asr/{main_lang}/{model_config}/"
        else:
            url = f"{fallback_url}/asr/{main_lang}/{model_config}/"
        transcription = session.post(
            url,
            data=np.asarray(data).tobytes(),
        )
        increase_queue(target_port)
        return transcription, target_port
