model=OpenAssistant/falcon-7b-sft-mix-2000
quantize=gptq

volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model #--quantize=$quantize