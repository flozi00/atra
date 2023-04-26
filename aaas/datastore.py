import hashlib
import os
import random
import time
from functools import lru_cache
from typing import Optional

import datasets
import fsspec
from sqlmodel import Field, Session, SQLModel, create_engine, select
from tqdm.auto import tqdm

from aaas.statics import CACHE_SIZE, INPROGRESS, TODO

db_backend = os.getenv("DBBACKEND", "sqlite:///database.db")
ftp_backend = os.getenv("FTPBACKEND")
try:
    ftp_user = ftp_backend.split("@")[0]
    ftp_pass = ftp_backend.split("@")[1]
    ftp_server = ftp_backend.split("@")[2]
except Exception:
    ftp_user, ftp_pass, ftp_server = None, None, None
build_dataset = os.getenv("BUILDDATASET", "false") == "true"


class QueueData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    metas: str
    transcript: str = Field(max_length=4096)
    model_config: str
    hash: str = Field(unique=True)
    langs: str = ""
    priority: int = 0
    votings: int = 0


engine = create_engine(db_backend, pool_recycle=3600, pool_pre_ping=True)
SQLModel.metadata.create_all(engine)


async def add_to_queue(audio_batch, hashes, master, model_config):
    # Create a session to the database
    with Session(engine) as session:
        # Loop over all audio files in the batch
        for x in range(len(audio_batch)):
            # Get the audio data
            audio_data = audio_batch[x]
            hs = hashes[x]
            # Get the timestamps for the current audio file
            time_dict = master[x]
            timesstamps = f"{time_dict['start']},{time_dict['end']}"
            # Get the entry from the database. If there is no entry, it returns None
            entry = get_transkript(hs)
            # If there is no entry in the database
            if entry is None:
                # Create a new entry
                entry = QueueData(
                    metas=timesstamps,
                    transcript=TODO,
                    model_config=model_config,
                    hash=hs,
                )
                # Add the audio data to the database
                set_data_to_hash(hs, audio_data)
                # Add the new entry to the session
                session.add(entry)
                # Commit the changes to the database
                session.commit()


def get_transkript(hs: str) -> QueueData:
    """Get a transkript from the database by its hash
    Args:
        hs (str): The hash of the transkript

    Returns:
        QueueData: The transkript from the database in a QueueData object
    """
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()

    return transkript


def get_transkript_batch(hs: str) -> list:
    """Get a transkript from the database by its hash in a batch

    Args:
        hs (str): The hashes of the transkripts separated by a comma

    Returns:
        _type_: The transkripts from the database in a QueueData object in a list
    """
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash.in_(hs.split(",")))
        transkript = session.exec(statement).all()

    return transkript


def get_tasks_queue() -> QueueData:
    """Get a random item from the queue

    Returns:
        QueueData: A random item from the queue
    """
    is_reclamation = False
    with Session(engine) as session:

        def get_queue(priority=1):
            statement = (
                select(QueueData)
                .where(QueueData.transcript == TODO)
                .where(QueueData.priority == priority)
            )
            todos = session.exec(statement).all()
            return todos

        # Get the first priority items
        todos = get_queue(1)
        # If no first priority items, get the second priority items
        if len(todos) == 0:
            todos = get_queue(priority=0)

        # Check if there are any items in the queue
        if len(todos) == 0:
            # Get all items that are in progress
            statement = select(QueueData).where(QueueData.transcript == INPROGRESS)
            todos = session.exec(statement).all()
            # Check if any items are in progress
            if len(todos) != 0:
                sample = random.choice(todos)
            else:
                # Get all items that have a negative voting score
                statement = select(QueueData).where(QueueData.votings < 0)
                todos = session.exec(statement).all()

        if len(todos) != 0:
            sample = todos[0]
        else:
            sample = False

    if sample is not False:
        is_reclamation = sample.votings < 0
        print("open todos ", len(todos))
    return sample, is_reclamation


def get_vote_queue(lang: str = None) -> QueueData:
    """Get the oldest item from the queue that has a
    voting score of 0 - 3 and is not in progress

    Args:
        lang (str, optional): language code string. Defaults to None.

    Returns:
        QueueData
    """
    with Session(engine) as session:
        statement = (
            select(QueueData)
            .where(QueueData.votings < 3)
            .where(QueueData.votings >= 0)
            .where(QueueData.transcript != TODO)
            .where(QueueData.transcript != INPROGRESS)
        )
        if lang is not None:
            statement = statement.where(QueueData.langs == lang)
        todos = session.exec(statement).all()
        if len(todos) == 0:
            sample = False
        else:
            sample = todos[0]

    return sample


def get_validated_dataset():
    dataset = []
    existing_dataset = datasets.load_dataset("flozi00/atra", split="train")
    hses = [x["hash"] for x in existing_dataset]
    with Session(engine) as session:
        statement = (
            select(QueueData)
            .where(QueueData.votings >= 100)
            .where(QueueData.transcript != TODO)
            .where(QueueData.transcript != INPROGRESS)
        )
        d_set = session.exec(statement).all()

    for x in tqdm(d_set):
        if x.hash not in hses:
            dataset.append(
                {
                    "lang": x.langs,
                    "text": x.transcript,
                    "hash": x.hash,
                    "bytes": get_data_from_hash(x.hash),
                }
            )
            remove_data_from_hash(x.hash)

    dataset = datasets.Dataset.from_list(dataset)
    dataset = datasets.concatenate_datasets([existing_dataset, dataset])

    dataset.push_to_hub("atra", private=True)


def set_transkript(
    hs: str, transcription: str, from_queue: bool = False, lang: str = None
):
    """Set the transcription of an audio file

    Args:
        hs (str): The hash of the audio file
        transcription (str): The transcription of the audio file
    """
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()
        if from_queue is True and "***" not in transkript.transcript:
            pass
        else:
            if transkript is not None and transkript.votings < 99:
                if transcription != transkript.transcript:
                    transkript.transcript = transcription
                    transkript.votings = 0
                    if lang is not None:
                        transkript.langs = lang
                    session.commit()
                    session.refresh(transkript)


def set_voting(hs: str, vote: str):
    """Set the voting of an audio file

    Args:
        hs (str): The hash of the audio file
        vote (str): The voting of the audio file, should be "good", "confirm" or "bad"
    """
    vote = 1 if vote == "good" else 100 if vote == "confirm" else -1
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            if transkript.votings < 99:
                transkript.votings = transkript.votings + vote
                if vote < 0:
                    transkript.transcript = TODO
                session.commit()
                session.refresh(transkript)


def set_in_progress(hs: str):
    """Set the transcription of an audio file to "INPROGRESS"

    Args:
        hs (str): The hash of the audio file
    """
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            transkript.transcript = INPROGRESS
            session.commit()
            session.refresh(transkript)


@lru_cache(maxsize=CACHE_SIZE)
def get_data_from_hash(hash: str) -> bytes:
    """Get the bytes of a file from the path which is the hash of the file

    Args:
        hash (str): hash of the file to be retrieved

    Returns:
        bytes: bytes of the file to be retrieved
    """
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


def set_data_to_hash(hs: str, bytes_data: bytes):
    """Store the bytes of a file to the path which is the hash of the file

    Args:
        data (str): hash of the file to be stored
        bytes_data (bytes): bytes of the file to be stored
    """
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
        with fs.open(f"data/{hs}", "wb") as f:
            f.write(bytes_data)
    except Exception as e:
        print(e)
        time.sleep(1)
        set_data_to_hash(hs, bytes_data)


def remove_data_from_hash(hs: str):
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
        fs.rm(f"data/{hs}")
    except Exception as e:
        print(e)


if build_dataset:
    get_validated_dataset()
