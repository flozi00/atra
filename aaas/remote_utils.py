from aaas.backend_utils import CPU_BACKENDS, GPU_BACKENDS, inference_only
import random

if inference_only == False:
    import yt_dlp as youtube_dl
    from shutil import move
    from requests_futures.sessions import FuturesSession
    import numpy as np

    session = FuturesSession()

    class FilenameCollectorPP(youtube_dl.postprocessor.common.PostProcessor):
        def __init__(self):
            super(FilenameCollectorPP, self).__init__(None)
            self.filenames = []
            self.tags = ""

        def run(self, information):
            self.filenames.append(information["filepath"])
            self.tags = " ".join(information["tags"][:3])
            return [], information

    def download_audio(url):
        options = {
            # "format": "best",
            # "postprocessors": [
            #    {
            #        "key": "FFmpegExtractAudio",
            #        "preferredcodec": "mp3",
            #        "preferredquality": "320",
            #    }
            # ],
            "outtmpl": "%(title)s.%(ext)s",
        }

        ydl = youtube_dl.YoutubeDL(options)
        filename_collector = FilenameCollectorPP()
        ydl.add_post_processor(filename_collector)
        ydl.download([url])

        fname, tags = filename_collector.filenames[0], filename_collector.tags
        move(fname, tags + "." + fname.split(".")[-1])
        fname = tags + "." + fname.split(".")[-1]

        return fname

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
