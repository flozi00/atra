# FROM nvcr.io/nvidia/pytorch:22.12-py3
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/libcudart.so' >> ~/.bashrc 


# CMD [ "jupyter", "notebook", "--allow-root", "--ip=\"0.0.0.0\"" ]