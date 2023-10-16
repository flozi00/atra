# for default german speech recognition model
```
docker run -d --gpus '"device=0"' -p 7862:7860  flozi00/atra:latest asrapp.py
```