# this script uses env vars and passes them to the python script
# it checks if the env vars exists before passing them to the python script
# 1. create base string for the command
# 2. check if env vars exists and add them to the command
# 3. run the command

cmd="python3 -m vllm.entrypoints.openai.api_server "

if [ -n "$VLLM_ADDITIONAL_ARGS" ]; then
  cmd="$cmd $VLLM_ADDITIONAL_ARGS"
fi

# 3. run the command
echo "Running command: $cmd"
$cmd
