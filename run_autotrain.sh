pip install uvicorn
pip install autotrain-advanced --upgrade
pip install --force-reinstall nvidia-ml-py nvitop
python3 -m uvicorn autotrain.app.app:app --host 0.0.0.0 --port 5001 --reload --workers 4
