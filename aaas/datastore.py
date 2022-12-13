from typing import Optional
from aaas.utils import timeit
from sqlmodel import Field, Session, SQLModel, create_engine, select
import soundfile as sf
import base64
import random
import os
import hashlib
from aaas.statics import TODO, INPROGRESS

db_backend = os.getenv("DBBACKEND", "sqlite:///database.db")


class AudioQueue(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    master: str
    data: str
    transcript: str
    main_lang: str
    model_config: str
    hs: str


engine = create_engine(db_backend)

SQLModel.metadata.create_all(engine)


@timeit
def add_audio(audio_batch, master, main_lang, model_config):
    path = str(random.randint(1, 999999999999999)) + ".wav"
    hashes = []
    with Session(engine) as session:
        for audio in audio_batch:
            sf.write(file=path, data=audio, samplerate=16000)

            with open(path, "rb") as bfile:
                audio_data = base64.b64encode(bfile.read()).decode("UTF-8")
            os.remove(path)
            hs = hashlib.sha256(audio_data.encode("utf-8")).hexdigest()
            hashes.append(hs)
            entry = AudioQueue(
                master=master,
                data=audio_data,
                transcript=TODO,
                main_lang=main_lang,
                model_config=model_config,
                hs=hs,
            )
            session.add(entry)

        session.commit()

    return hashes


@timeit
def get_transkript(data):
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.hs == data)
        transkript = session.exec(statement).first()

        return transkript.transcript


def get_audio_queue():
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.transcript == TODO)
        todos = session.exec(statement).all()

        if len(todos) != 0:
            sample = random.choice(todos)
        else:
            statement = select(AudioQueue).where(AudioQueue.transcript == INPROGRESS)
            todos = session.exec(statement).all()
            if len(todos) != 0:
                sample = random.choice(todos)
            else:
                sample = False
        return sample


def set_transkript(hs, transcription):
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.hs == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            transkript.transcript = transcription
            transkript.data = ""
            session.commit()
            session.refresh(transkript)


def set_in_progress(hs):
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.hs == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            transkript.transcript = INPROGRESS
            session.commit()
            session.refresh(transkript)


def delete_by_master(master):
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.master == master)
        results = session.exec(statement).all()
        for res in results:
            session.delete(res)

        session.commit()


@timeit
def delete_by_hashes(hashes):
    with Session(engine) as session:
        for hs in hashes:
            statement = select(AudioQueue).where(AudioQueue.hs == hs)
            res = session.exec(statement).first()
            session.delete(res)

        session.commit()
