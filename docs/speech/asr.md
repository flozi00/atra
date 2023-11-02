# for default german speech recognition model
```
docker run -d --pull always --gpus '"device=0"' -p 7862:7860  flozi00/atra:latest asrapp.py
```

## Environment variables

* ASR_MODEL - the model to use for speech recognition