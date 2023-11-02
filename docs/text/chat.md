# run tgi for chatbot
```
docker run --pull always --gpus all -d --shm-size 1g -p 8080:80 -v ./data:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --max-total-tokens 8192 --max-batch-prefill-tokens 8192 --max-input-length 5000 --model-id flozi00/Mistral-7B-german-assistant-v4-4bit-autogptq --quantize=gptq --cuda-memory-fraction 0.8
```

```
docker run --gpus all -p 8081:80 --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-0.2.2 --model-id intfloat/multilingual-e5-large
```

## Environment variables for chatapp

* LLM - the model to use for language model, either TGI endpoint or HF model
* EMBEDDER_HOST - the url for the text embedding inference server
* TYPESENSE_API_KEY - the api key for typesense
* TYPESENSE_HOST - the host ip or domain for typesense
* SERP_API_KEY - the api key for serper.dev


## Environment variables for train_assistant

* BASE_MODEL - the base model to use for training, huggingface hub id
* PEFT_MODEL - the model to use for peft, name of the target model
* DATASET_PATH - the path to the dataset to use for training, huggingface hub id
* DATASET_COLUMN - the column in the dataset to use for training, contains the text
* BATCH_SIZE - the batch size to use for training
* ACCUMULATION_STEPS - the gradient accumulation steps to use for training
* SEQ_LENGTH - the sequence length to use for training
* LORA_DEPTH - the depth of the lora model to use for training

### example docker command:

```
docker run -d --pull always --gpus '"device=0"' -e BASE_MODEL="HuggingFaceH4/zephyr-7b-alpha" -e PEFT_MODEL="Mistral-zephyr-german-assistant-v1" flozi00/atra:latest train_assistant.py
```