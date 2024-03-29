pip install uvicorn
pip install autotrain-advanced --upgrade
python3 -m uvicorn autotrain.app:app --host 0.0.0.0 --port 5001 --reload --workers 4