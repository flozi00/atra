# FROM nvcr.io/nvidia/pytorch:22.12-py3
FROM pytorch/pytorch

RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/libcudart.so' >> ~/.bashrc 

RUN /opt/conda/bin/pip install bitsandbytes 
RUN /opt/conda/bin/pip install jupyter jiwer
RUN /opt/conda/bin/pip install transformers datasets
RUN /opt/conda/bin/pip install accelerate

# CMD [ "jupyter", "notebook", "--allow-root", "--ip=\"0.0.0.0\"" ]