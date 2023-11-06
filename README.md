# ATRA - Automatic Transcription, Recognition and Assistance

This project focuses on providing speech recognition and intelligent assistants across industries and languages.

The project is being developed on a private basis with support of Primeline Solutions who is the main sponsor of my research.
Currently I'm focusing on languages used in the German area, but I'm happy about any support for other languages.

## How to install
The fastest way to run is using docker:


# for default german speech recognition model
```
docker run -d --gpus '"device=0"' -p 7862:7860  flozi00/atra:latest asrapp.py
```

# Stable diffusion dream shaper model
```
docker run -d --gpus '"device=0"' -p 7861:7860  flozi00/atra:latest sdapp.py
```