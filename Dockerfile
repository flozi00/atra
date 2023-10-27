FROM huggingface/transformers-pytorch-gpu:latest

RUN apt update && apt-get install ffmpeg -y

COPY . /atra-server                                                            
WORKDIR /atra-server

RUN pip uninstall transformers -y
RUN pip uninstall transformers -y
RUN pip install -r requirements.txt
RUN pip install transformers --upgrade
RUN pip install flash-attn --no-build-isolation --upgrade
RUN playwright install

RUN chmod +x ./entrypoint.sh

ENTRYPOINT [ "/atra-server/entrypoint.sh" ]

CMD [ "sdapp.py" ]
