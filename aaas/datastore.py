import base64
import hashlib
import os
import random
from typing import Optional

import soundfile as sf
from sqlmodel import Field, Session, SQLModel, create_engine, select

from aaas.statics import INPROGRESS, TODO
from aaas.utils import timeit

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
        for x in range(len(audio_batch)):
            audio = audio_batch[x]
            time_dict = master[x]
            times = f"{time_dict['start']},{time_dict['end']}"
            sf.write(file=path, data=audio, samplerate=16000)

            with open(path, "rb") as bfile:
                audio_data = base64.b64encode(bfile.read()).decode("UTF-8")
            os.remove(path)
            hs = hashlib.sha256(
                audio_data.encode("utf-8")
                + f"{main_lang}, {model_config}, {times}".encode("utf-8")
            ).hexdigest()
            hashes.append(hs)
            entry = AudioQueue(
                master=times,
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
def get_transkript(hs):
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.hs == hs)
        transkript = session.exec(statement).first()

    return transkript


@timeit
def get_all_transkripts():
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.data == "")
        transkripts = session.exec(statement).all()

    return transkripts


@timeit
def get_audio_queue():
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.transcript == TODO).limit(3)
        todos = session.exec(statement).all()

        if len(todos) != 0:
            sample = random.choice(todos)
        else:
            statement = (
                select(AudioQueue).where(AudioQueue.transcript == INPROGRESS).limit(3)
            )
            todos = session.exec(statement).all()
            if len(todos) != 0:
                sample = random.choice(todos)
            else:
                sample = False
    return sample


@timeit
def set_transkript(hs, transcription):
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.hs == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            transkript.transcript = transcription
            transkript.data = ""
            session.commit()
            session.refresh(transkript)


@timeit
def set_in_progress(hs):
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.hs == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            transkript.transcript = INPROGRESS
            session.commit()
            session.refresh(transkript)


@timeit
def delete_by_hashes(hashes):
    with Session(engine) as session:
        for hs in hashes:
            statement = select(AudioQueue).where(AudioQueue.hs == hs)
            res = session.exec(statement).first()
            session.delete(res)

        session.commit()
