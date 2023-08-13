from pathlib import Path
from os import environ
from uuid import uuid4


class ConfigService:
    APP_DATA_PATH = '/tmp/fonty'

    def __init__(self):
        self.store = Path(environ.get('FONTY_DATA_PATH', self.APP_DATA_PATH))
        self.store.mkdir(parents=True, exist_ok=True)
        self.uuid = uuid4
