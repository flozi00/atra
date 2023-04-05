FROM python:3.9.15-slim

RUN apt update && apt-get install ffmpeg -y

COPY . /asr-server                                                            
WORKDIR /asr-server

RUN pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir && pip install -r requirements.txt --no-cache-dir

ENV DBBACKEND=sqlite:///database.db
ENV BUILDDATASET=false
ENV PORT=7860
ENV ADMINMODE=false

EXPOSE 7860

CMD exec python app.py
