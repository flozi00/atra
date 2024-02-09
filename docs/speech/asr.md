# for default german speech recognition model
```
docker run -d --pull always --gpus '"device=0"' -p 7862:7860 --env-file .env flozi00/atra:latest asrapp
```

## Environment variables

* ASR_MODEL - the model to use for speech recognition