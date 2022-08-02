FROM python:3.9

COPY ./requirements.txt /asr/requirements.txt
COPY ./app.py /asr/app.py 
COPY ./test.py /asr/test.py       
COPY ./2g-german-kenlm /asr/2g-german-kenlm                                                     
WORKDIR /asr   

# Install requirements and db stuff
RUN apt-get update && \
    apt-get upgrade -y && apt-get install ffmpeg build-essential cmake -y

RUN pip install -r requirements.txt
RUN python test.py

EXPOSE 7860
CMD exec python app.py