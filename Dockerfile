FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt update && apt-get install ffmpeg -y

COPY . /atra-server                                                            
WORKDIR /atra-server

RUN pip install -r requirements.txt
RUN pip install transformers --upgrade
RUN pip install flash-attn --no-build-isolation --upgrade
RUN playwright install

RUN chmod +x ./entrypoint.sh

ENTRYPOINT [ "/atra-server/entrypoint.sh" ]

CMD [ "sdapp.py" ]
