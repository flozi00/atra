import yt_dlp as youtube_dl
from shutil import move
from requests_futures.sessions import FuturesSession
import numpy as np
import os

backend = os.getenv("inference_backend", "http://127.0.0.1:7860")

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
    move(fname, tags + " " + fname)
    fname = tags + " " + fname

    return fname


def remote_inference(main_lang, model_config, data):
    transcription = session.post(
        f"{backend}/asr/{main_lang}/{model_config}/", data=np.asarray(data).tobytes()
    )
    return transcription
