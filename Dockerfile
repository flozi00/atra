FROM python:3.9

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs ffmpeg -y

COPY . /asr-server                                                            
WORKDIR /asr-server

RUN pip install -r requirements.txt && git submodule init && git submodule update && python test.py

CMD exec python app.py