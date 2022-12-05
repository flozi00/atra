FROM python:3.9.15-slim

RUN apt update && apt-get install ffmpeg -y

COPY . /asr-server                                                            
WORKDIR /asr-server

RUN pip install torch>=1.13.0 --extra-index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

EXPOSE 7860

CMD exec python app.py
