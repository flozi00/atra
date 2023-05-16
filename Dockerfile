FROM python:3.9.15-slim

RUN apt update && apt-get install ffmpeg -y

COPY . /asr-server                                                            
WORKDIR /asr-server

RUN pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir && pip install -r requirements.txt --no-cache-dir

ENV DBBACKEND=sqlite:///database.db
ENV PORT=7860

EXPOSE 7860

CMD exec python app.py
