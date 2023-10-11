# ATRA - Automatic Transcription, Recognition and Assistance

This project focuses on providing speech recognition and intelligent assistants across industries and languages.

The project is being developed on a private basis with support Primeline Solutions who is the main sponsor of my research.
Currently I'm focusing on languages used in the German area, but I'm happy about any support for other languages.

An example of training large ASR models on small hardware take a look at the simplepeft project https://github.com/flozi00/simplepeft 

## How to install
The fastest way to run is using docker:
example given:
tiny model

```
docker run --gpus '"device=0"' -e ASR_MODEL="openai/whisper-tiny" -p 7868:7860  flozi00/atra:latest asrapp.py
```

# for default german speech recognition model
```
docker run -d --gpus '"device=0"' -p 7862:7860  flozi00/atra:latest asrapp.py
```

# Stable diffusion dream shaper model
```
docker run -d --gpus '"device=0"' -p 7861:7860  flozi00/atra:latest sdapp.py
```

# Jupyter notebook development
```
docker run -d --gpus '"device=0"' -p 8888:8888  flozi00/atra:latest devapp.py
```
