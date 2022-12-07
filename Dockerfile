FROM python:3.9.15-slim

RUN apt update && apt-get install ffmpeg -y

COPY . /asr-server                                                            
WORKDIR /asr-server

RUN pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir && pip install -r requirements.txt --no-cache-dir

EXPOSE 7860

CMD exec python app.py
