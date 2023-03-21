import hashlib
import os
import random
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from aaas.statics import CACHE_SIZE, INPROGRESS, TODO
from aaas.utils import timeit
import fsspec
import time
from functools import lru_cache
import datasets
from tqdm.auto import tqdm

db_backend = os.getenv("DBBACKEND", "sqlite:///database.db")
ftp_backend = os.getenv("FTPBACKEND")
ftp_user = ftp_backend.split("@")[0]
ftp_pass = ftp_backend.split("@")[1]
ftp_server = ftp_backend.split("@")[2]
build_dataset = os.getenv("BUILDDATASET", "false") == "true"


class QueueData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    metas: str
    transcript: str = Field(max_length=4096)
    langs: str
    model_config: str
    hash: str
    priority: int = 0
    votings: int = 0


engine = create_engine(db_backend, pool_recycle=3600, pool_pre_ping=True)
SQLModel.metadata.create_all(engine)


@timeit
def add_to_queue(
    audio_batch, master, main_lang, model_config, times=None, file_format="wav"
):
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
            hs = f"{hs}.{file_format}"
            hashes.append(hs)
            entry = get_transkript(hs)
            if entry is None:
                entry = QueueData(
                    metas=timesstamps,
                    transcript=TODO,
                    langs=main_lang,
                    model_config=model_config,
                    hash=hs,
                )
                set_data_to_hash(entry, audio_data)
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
def get_transkript_batch(hs):
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash.in_(hs.split(",")))
        transkript = session.exec(statement).all()

    return transkript


def get_tasks_queue():
    with Session(engine) as session:

        def get_queue(priority=1):
            statement = (
                select(QueueData)
                .where(QueueData.transcript == TODO)
                .where(QueueData.priority == priority)
            )
            todos = session.exec(statement).all()
            return todos

        todos = get_queue(1)
        if len(todos) == 0:
            todos = get_queue(priority=0)

        if len(todos) != 0:
            sample = random.choice(todos)
        else:
            statement = select(QueueData).where(QueueData.transcript == INPROGRESS)
            todos = session.exec(statement).all()
            if len(todos) != 0:
                sample = random.choice(todos)
            else:
                statement = select(QueueData).where(QueueData.votings < 0)
                todos = session.exec(statement).all()
                if len(todos) != 0:
                    sample = random.choice(todos)
                else:
                    sample = False
    return sample


def get_vote_queue():
    with Session(engine) as session:
        statement = (
            select(QueueData)
            .where(QueueData.votings < 3)
            .where(QueueData.votings >= 0)
            .where(QueueData.transcript != TODO)
            .where(QueueData.transcript != INPROGRESS)
        )
        todos = session.exec(statement).all()
        sample = todos[-1]

    return sample


def get_validated_dataset():
    dataset = []
    with Session(engine) as session:
        statement = (
            select(QueueData)
            .where(QueueData.votings >= 100)
            .where(QueueData.transcript != TODO)
            .where(QueueData.transcript != INPROGRESS)
        )
        d_set = session.exec(statement).all()

    for x in tqdm(d_set):
        dataset.append(
            {
                "lang": x.langs,
                "text": x.transcript,
                "hash": x.hash,
                "bytes": get_data_from_hash(x.hash),
            }
        )

    dataset = datasets.Dataset.from_list(dataset)

    dataset.push_to_hub("atra", private=True)


@timeit
def set_transkript(hs, transcription):
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()
        if transkript is not None and transkript.votings < 99:
            transkript.transcript = transcription
            transkript.votings = 0
            session.commit()
            session.refresh(transkript)


@timeit
def set_voting(hs, vote):
    vote = 1 if vote == "good" else 100 if vote == "confirm" else -1
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            if transkript.votings < 99:
                transkript.votings = transkript.votings + vote
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


@timeit
@lru_cache(maxsize=CACHE_SIZE)
def get_data_from_hash(hash: str):
    if ftp_backend is not None:
        fs = fsspec.filesystem(
            "ftp",
            host=ftp_server,
            username=ftp_user,
            password=ftp_pass,
            port=21,
            block_size=2**20,
        )
    else:
        fs = fsspec.filesystem("file")
    with fs.open(f"data/{hash}", "rb") as f:
        bytes_data = f.read()

    return bytes_data


@timeit
def set_data_to_hash(data: QueueData, bytes_data: bytes):
    if ftp_backend is not None:
        fs = fsspec.filesystem(
            "ftp",
            host=ftp_server,
            username=ftp_user,
            password=ftp_pass,
            port=21,
            block_size=2**20,
        )
    else:
        fs = fsspec.filesystem("file")
    try:
        with fs.open(f"data/{data.hash}", "wb") as f:
            f.write(bytes_data)
    except Exception as e:
        print(e)
        time.sleep(1)
        set_data_to_hash(data, bytes_data)


if build_dataset:
    get_validated_dataset()
