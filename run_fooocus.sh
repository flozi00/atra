cd /

git clone https://github.com/lllyasviel/Fooocus.git /Fooocus

cd /Fooocus
git pull
git checkout "1c999be8c8134fe01a75723ea933858435856950"
git pull

pip install torchsde

# echo All files are downloaded, now running the server
echo "All files are downloaded, now running the server"

rm /Fooocus/config.txt

cp -f /atra-server/atra/utilities/config.txt /Fooocus/config.txt

cmd="python3 /Fooocus/entry_with_update.py --listen --always-gpu --vae-in-fp16 --unet-in-fp8-e4m3fn --clip-in-fp8-e4m3fn"

if [ -n "$SERVER_PORT" ]; then
  cmd="$cmd --port $SERVER_PORT"
fi

echo "Running command: $cmd"
$cmd
