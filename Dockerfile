FROM nvcr.io/nvidia/pytorch:24.02-py3

RUN apt update && apt-get install ffmpeg -y

COPY . /atra-server                                                            
WORKDIR /atra-server

RUN pip uninstall transformer-engine -y
RUN pip install -r requirements.txt
RUN pip install torch-tensorrt tensorrt --upgrade
RUN pip install flash-attn --no-build-isolation --upgrade

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["/bin/bash"]

CMD [ "run_asr.sh" ]
