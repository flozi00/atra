from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select
import soundfile as sf
import base64
import random
import os

db_backend = os.getenv("DBBACKEND", "sqlite:///database.db")

class AudioQueue(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    master: str
    data: str
    transcript: str
    main_lang: str
    model_config: str
    
engine = create_engine(db_backend)

SQLModel.metadata.create_all(engine)


def add_audio(audio, master, main_lang, model_config):
    path = str(random.randint(1,999999999999999)) + ".wav"

    sf.write(file=path, data=audio, samplerate=16000)
    
    with open(path, "rb") as bfile:
        audio_data = base64.b64encode(bfile.read()).decode('UTF-8')
    os.remove(path)
    
    entry = AudioQueue(master=master, data=audio_data, transcript="***TODO***", main_lang=main_lang, model_config=model_config)
    
    with Session(engine) as session:
        session.add(entry)
        session.commit()
    
    return audio_data

def transkripts_done(master) -> bool:
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.master == master).where(AudioQueue.transcript == "***TODO***")
        transkripts = session.exec(statement).all()
        return len(transkripts) == 0
    
def get_transkript(data) -> list:
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.data == data)
        transkript = session.exec(statement).first()
        
        return transkript.transcript
    
def get_audio_queue():
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.transcript == "***TODO***")
        todos = session.exec(statement).all()
        
        if(len(todos) != 0):
            sample = random.choice(todos)
        else:
            sample = False    
        return sample
        
def set_transkript(data, transcription) -> list:
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.data == data)
        transkript = session.exec(statement).first()
        transkript.transcript = transcription
        session.commit()
        session.refresh(transkript)
        
def delete_by_master(master):
    with Session(engine) as session:
        statement = select(AudioQueue).where(AudioQueue.master == master)
        results = session.exec(statement).all()
        for res in results:
            session.delete(res)
            
        session.commit()