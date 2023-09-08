from pymongo import MongoClient

# *----------------------------------------------------------------------------* Remote DB

REMOTE_DATABASE = "fonty"
REMOTE_COLLECTION = "fonts"
REMOTE_CONFIG = {
    "host": "mongodb+srv://fonty.hquocfa.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority",
    "tlsCertificateKeyFile": "auth/X509-cert-6658948524510096383.pem",
    "tls": True,
}


class RemoteClient:
    """Provides an access to processed fonts stored in MongoDB Atlas."""

    def __init__(self):
        self.client = MongoClient(**REMOTE_CONFIG)
        self.col = self.client[REMOTE_DATABASE][REMOTE_COLLECTION]


# *----------------------------------------------------------------------------* Local DB

LOCAL_DATABASE = "fonty"
LOCAL_COLLECTION = "data"
LOCAL_CONFIG = {
    # "host": "192.168.1.11", # local network
    "host": "169.254.130.43", # direct wired connection (link-local)
    "port": 27017,
}


class LocalClient:
    """Provides an access to local MongoDB used for caching rendered glyphs."""

    def __init__(self):
        self.client = MongoClient(**LOCAL_CONFIG)
        self.col = self.client[LOCAL_DATABASE][LOCAL_COLLECTION]