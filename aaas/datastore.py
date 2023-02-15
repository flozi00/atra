import hashlib
import os
import random
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from aaas.statics import INPROGRESS, TODO
from aaas.utils import timeit

db_backend = os.getenv("DBBACKEND", "sqlite:///database.db")


class QueueData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    metas: str
    data: bytes = Field(max_length=(2**32) - 1)
    transcript: str = Field(max_length=4096)
    langs: str
    model_config: str
    hash: str
    priority: int = 0


engine = create_engine(db_backend, pool_recycle=3600, pool_pre_ping=True)
SQLModel.metadata.create_all(engine)


@timeit
def add_to_queue(audio_batch, master, main_lang, model_config, times=None):
    hashes = []
    with Session(engine) as session:
        for x in range(len(audio_batch)):
            audio_data = audio_batch[x]
            if times == None:
                time_dict = master[x]
                timesstamps = f"{time_dict['start']},{time_dict['end']}"
            else:
                timesstamps = times

            hs = hashlib.sha256(
                f"{audio_data} {main_lang}, {model_config}, {timesstamps}".encode(
                    "utf-8"
                )
            ).hexdigest()
            hashes.append(hs)
            entry = get_transkript(hs)
            if entry is None:
                entry = QueueData(
                    metas=timesstamps,
                    data=audio_data,
                    transcript=TODO,
                    langs=main_lang,
                    model_config=model_config,
                    hash=hs,
                )
                session.add(entry)
                session.commit()

    return hashes


@timeit
def get_transkript(hs):
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()

    return transkript


@timeit
def get_tasks_queue():
    with Session(engine) as session:

        def get_queue(priority=1):
            statement = (
                select(QueueData)
                .where(QueueData.transcript == TODO)
                .where(QueueData.priority == priority)
                .limit(3)
            )
            todos = session.exec(statement).all()
            return todos

        todos = get_queue(1)
        if len(todos) == 0:
            todos = get_queue(priority=0)

        if len(todos) != 0:
            sample = random.choice(todos)
        else:
            statement = (
                select(QueueData).where(QueueData.transcript == INPROGRESS).limit(3)
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
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            transkript.transcript = transcription
            session.commit()
            session.refresh(transkript)


@timeit
def set_in_progress(hs):
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            transkript.transcript = INPROGRESS
            session.commit()
            session.refresh(transkript)
