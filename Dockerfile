FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt update && apt-get install ffmpeg -y

ARG PYTORCH='2.1.0'
# (not always a valid torch version)
ARG INTEL_TORCH_EXT='1.11.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu118'

RUN [ ${#PYTORCH} -gt 0 -a "$PYTORCH" != "pre" ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; echo "export VERSION='$VERSION'" >> ~/.profile
RUN echo torch=$VERSION
# `torchvision` and `torchaudio` should be installed along with `torch`, especially for nightly build.
# Currently, let's just use their latest releases (when `torch` is installed with a release version)
# TODO: We might need to specify proper versions that work with a specific torch version (especially for past CI).
RUN [ "$PYTORCH" != "pre" ] && python3 -m pip install --no-cache-dir -U $VERSION torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA || python3 -m pip install --no-cache-dir -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/$CUDA


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