import os
import random
from typing import Optional, Union

from sqlmodel import Field, Session, SQLModel, create_engine, select

from atra.statics import TODO, TASK_MAPPING
from atra.utils import timeit

db_backend = os.getenv("DBBACKEND", "sqlite:///database.db")

if db_backend.startswith("sqlite"):
    if os.path.isfile(db_backend.replace("sqlite:///", "")):
        os.remove(db_backend.replace("sqlite:///", ""))

class QueueData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    metas: str
    transcript: str = Field(max_length=4096)
    hash: str = Field(unique=True)
    file_object: bytes


engine = create_engine(db_backend, pool_recycle=3600, pool_pre_ping=True)
SQLModel.metadata.create_all(engine)

@timeit
def add_to_queue(audio_batch, hashes, times_list):
    # Create a session to the database
    with Session(engine) as session:
        # Loop over all audio files in the batch
        for x in range(len(audio_batch)):
            timesstamps = None
            # Get the audio data
            audio_data = audio_batch[x]
            hs = hashes[x]
            # Get the timestamps for the current audio file
            time_dict = times_list[x]
            # in case its asr with timestamps
            for task in list(TASK_MAPPING.keys()):
                if (
                    TASK_MAPPING[task][0] in time_dict
                    and TASK_MAPPING[task][1] in time_dict
                ):
                    timesstamps = ""
                    for task_key in TASK_MAPPING[task]:
                        timesstamps += f"{time_dict[task_key]},"
                    timesstamps += f"{task}"

            if timesstamps is not None:
                # Create a new entry
                try:
                    entry = QueueData(
                        metas=timesstamps,
                        transcript=TODO,
                        hash=hs,
                        file_object=audio_data,
                    )
                    # Add the new entry to the session
                    session.add(entry)
                    # Commit the changes to the database
                    session.commit()
                except Exception as e:
                    print(e)
                    session.rollback()

def get_transkript_batch(hs: str) -> list:
    """Get a transkript from the database by its hash in a batch

    Args:
        hs (str): The hashes of the transkripts separated by a comma

    Returns:
        _type_: The transkripts from the database in a QueueData object in a list
    """
    with Session(engine) as session:
        statement = select(QueueData).where(
            QueueData.hash.in_(hs.split(","))) # type: ignore
        transkript = session.exec(statement).all()

    return transkript

def get_tasks_queue() -> Union[QueueData, None]:
    """Get a random item from the queue

    Returns:
        QueueData: A random item from the queue
    """
    with Session(engine) as session:
        sample = None
        statement = select(QueueData).where(QueueData.transcript == TODO)
        todos = session.exec(statement).all()

        if len(todos) != 0:
            sample = random.choice(todos)

    return sample

@timeit
def set_transkript(hs: str, transcription: str):
    """Set the transcription of an audio file

    Args:
        hs (str): The hash of the audio file
        transcription (str): The transcription of the audio file
    """
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            if transcription != transkript.transcript:
                transkript.transcript = transcription
                transkript.file_object = bytes()
                session.commit()

@timeit
def delete_by_hash(hs: str):
    with Session(engine) as session:
        statement = select(QueueData).where(QueueData.hash == hs)
        transkript = session.exec(statement).first()
        if transkript is not None:
            session.delete(transkript)
            session.commit()
