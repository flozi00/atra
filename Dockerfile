FROM python:3.9

RUN apt update && apt-get install ffmpeg imagemagick imagemagick-doc -y

COPY . /asr-server                                                            
WORKDIR /asr-server

RUN pip install -r requirements.txt

EXPOSE 7860

CMD exec python app.py