FROM ubuntu:latest

RUN apt update && apt-get install ffmpeg imagemagick imagemagick-doc python3 python3-pip python-is-python3 -y

COPY . /asr-server                                                            
WORKDIR /asr-server

COPY policy.xml  /etc/ImageMagick-6/policy.xml

RUN pip install -r requirements.txt && pip install intel_extension_for_pytorch

EXPOSE 7860

CMD exec python app.py