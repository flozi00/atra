# ASR as a service

This project focuses on providing speech recognition across industries and languages.

The project is being developed on a private basis with support from A\\Ware and Reibke.
Currently I'm focusing on languages used in the German area, but I'm happy about any support for other languages.

With the exception of the audio dataset and training script (at the moment, coming in future), all code, used models and datasets are released under open source license.

## How to install
Install the ffmpeg package for loading the audio files and python3 to run the code

clone this repository and download the submodule with the languagemodel too.
Probably you will need git-lfs installed.
Update the whole system using "git pull --recurse-submodules"

With both packages installed you can just run "pip install -r requirements.txt" to install the python librarys and then start the server with gradio UI using "python app.py"
