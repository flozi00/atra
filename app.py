from atra.gradio_utils.asr import build_asr_ui
from atra.gradio_utils.sd import build_diffusion_ui
import torch.multiprocessing as mp

if __name__ == "__main__":
    build_diffusion_ui()
    """
    mp.set_start_method("spawn")
    UIs = [build_diffusion_ui, build_asr_ui]
    processes = [mp.Process(target=ui) for ui in UIs]
    # start new processes
    for child in processes:
        child.start()
    # wait for processes to finish
    for child in processes:
        child.join()
    """
