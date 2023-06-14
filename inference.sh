model=OpenAssistant/falcon-7b-sft-mix-2000
num_shard=1

volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:0.8 --model-id $model --num-shard $num_shard