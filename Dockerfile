FROM ubuntu:18.04

RUN apt update && apt-get install ffmpeg imagemagick imagemagick-doc python3 python3-pip sshpass -y

COPY . /asr-server                                                            
WORKDIR /asr-server

COPY policy.xml  /etc/ImageMagick-6/policy.xml

RUN pip3 install --upgrade pip
RUN pip3 install torch>=1.13.0 --extra-index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

EXPOSE 7860

CMD exec python3 app.py
