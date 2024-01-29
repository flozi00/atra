# Stable diffusion dream shaper model
```
docker run -d --pull always --gpus '"device=0"' -p 7861:7860 --env-file .env flozi00/atra:latest sdapp.py
```