import hashlib
import os
import random
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from aaas.statics import INPROGRESS, TODO
from aaas.utils import timeit

db_backend = os.getenv("DBBACKEND", "sqlite:///database.db")


class AudioData(SQLModel, table=True):
    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamps: str
    data: bytes
    transcript: str
    main_lang: str
    model_config: str
    hs: str
    priority: int = 0


@timeit
def add_audio(audio_batch, master, main_lang, model_config):
    engine = create_engine(db_backend)
    SQLModel.metadata.create_all(engine)

    hashes = []
    with Session(engine) as session:
        for x in range(len(audio_batch)):
            audio = audio_batch[x]
            time_dict = master[x]
            times = f"{time_dict['start']},{time_dict['end']}"

            audio_data = audio.tobytes()
            hs = hashlib.sha256(
                f"{audio_data} {main_lang}, {model_config}, {times}".encode("utf-8")
            ).hexdigest()
            hashes.append(hs)
            if get_transkript(hs) is None:
                entry = AudioData(
                    timestamps=times,
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
    engine = create_engine(db_backend)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        statement = select(AudioData).where(AudioData.hs == hs)
        transkript = session.exec(statement).first()

    return transkript


@timeit
def get_audio_queue():
    engine = create_engine(db_backend)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:

        def get_queue(priority=10):
            statement = (
                select(AudioData)
                .where(AudioData.transcript == TODO)
                .where(AudioData.priority == priority)
                .limit(3)
            )
            todos = session.exec(statement).all()
            if len(todos) == 0:
                todos = get_queue(priority=priority - 1)
            return todos

        todos = get_queue()

        if len(todos) != 0:
            sample = random.choice(todos)
        else:
            statement = (
                select(AudioData).where(AudioData.transcript == INPROGRESS).limit(3)
            )
            todos = session.exec(statement).all()
            if len(todos) != 0:
                sample = random.choice(todos)
            else:
                sample = False
    return sample


@timeit
def set_transkript(hs, transcription):
    engine = create_engine(db_backend)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        statement = select(AudioData).where(AudioData.hs == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            transkript.transcript = transcription
            session.commit()
            session.refresh(transkript)


@timeit
def set_in_progress(hs):
    engine = create_engine(db_backend)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        statement = select(AudioData).where(AudioData.hs == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            transkript.transcript = INPROGRESS
            session.commit()
            session.refresh(transkript)


@timeit
def delete_by_hashes(hashes):
    engine = create_engine(db_backend)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        for hs in hashes:
            statement = select(AudioData).where(AudioData.hs == hs)
            res = session.exec(statement).first()
            session.delete(res)

        session.commit()
