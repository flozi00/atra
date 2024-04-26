FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt update && apt-get install ffmpeg -y

COPY . /atra-server                                                            
WORKDIR /atra-server

RUN pip uninstall transformer-engine -y
RUN pip install -r requirements.txt

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["/bin/bash"]

CMD [ "run_asr.sh" ]
