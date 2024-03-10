# this script uses env vars and passes them to the python script
# it checks if the env vars exists before passing them to the python script
# 1. create base string for the command
# 2. check if env vars exists and add them to the command
# 3. run the command

# 1. create base string for the command
cmd="python3 -m vllm.entrypoints.api_server"

# 2. check if env vars exists and add them to the command
# check for --host, --port, --quantization, --model, --kv-cache-dtype, --gpu-memory-utilization

if [ -n "$VLLM_HOST" ]; then
  cmd="$cmd --host $VLLM_HOST"
fi

if [ -n "$VLLM_PORT" ]; then
  cmd="$cmd --port $VLLM_PORT"
fi

if [ -n "$VLLM_QUANTIZATION" ]; then
  cmd="$cmd --quantization $VLLM_QUANTIZATION"
fi

if [ -n "$VLLM_MODEL" ]; then
  cmd="$cmd --model $VLLM_MODEL"
fi

if [ -n "$VLLM_KV_CACHE_DTYPE" ]; then
  cmd="$cmd --kv-cache-dtype $VLLM_KV_CACHE_DTYPE"
fi

if [ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]; then
  cmd="$cmd --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION"
fi

# 3. run the command
echo "Running command: $cmd"
$cmd