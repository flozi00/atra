FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt update && apt-get install ffmpeg -y

COPY . /atra-server                                                            
WORKDIR /atra-server

#RUN pip install flash-attn --no-build-isolation --upgrade
RUN pip uninstall transformer-engine -y
RUN pip install --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator
RUN pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
RUN pip install -r requirements.txt
RUN pip install transformers diffusers --upgrade
RUN playwright install

RUN chmod +x ./entrypoint.sh

ENTRYPOINT [ "/atra-server/entrypoint.sh" ]

CMD [ "sdapp" ]
