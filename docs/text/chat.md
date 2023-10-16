# run tgi for chatbot
```
docker run --pull always --gpus all -d --shm-size 1g -p 8080:80 -v ./data:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --max-total-tokens 8192 --max-batch-prefill-tokens 8192 --max-input-length 7600 --model-id flozi00/Mistral-7B-german-assistant-v2-4bit-autogptq --quantize=gptq --cuda-memory-fraction 1
```