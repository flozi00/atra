cd /

git clone https://github.com/lllyasviel/Fooocus.git /Fooocus

cd /Fooocus
git pull
git checkout "v2.4.3"
git pull

pip install torchsde

# echo All files are downloaded, now running the server
echo "All files are downloaded, now running the server"

rm /Fooocus/config.txt

cp -f /atra-server/atra/utilities/config.txt /Fooocus/config.txt

cmd="python3 /Fooocus/entry_with_update.py --listen --vae-in-fp16"

if [ -n "$SERVER_PORT" ]; then
  cmd="$cmd --port $SERVER_PORT"
fi

if [ -n "$FP8" ]; then
  cmd="$cmd --unet-in-fp8-e4m3fn --clip-in-fp8-e4m3fn"
fi

echo "Running command: $cmd"
$cmd
