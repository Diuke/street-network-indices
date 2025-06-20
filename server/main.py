import os
from typing import Union
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OSM Street Network Downloader"
    model_config = SettingsConfigDict(env_file=".env")

    data_folder:str = os.getenv("DATA_FOLDER", "/home/user/Desktop/JP/street-network-indices/data/geoinf")
    print(f"DATA_FOLDER location: {data_folder}")

settings = Settings()
app = FastAPI()

@app.get("/")
def read_root():
    print(settings.data_folder)
    return {"Hello": "World"}

@app.get("/download-networks")
def read_root(drive:bool = False, pedestrian:bool = False, cycling:bool = False, public_transport:bool = False):
    print(settings.data_folder)
    return {"Hello": "World"}
    


