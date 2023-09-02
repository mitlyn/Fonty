from pymongo import MongoClient

# *----------------------------------------------------------------------------* MongoDB Config

MONGO_CONFIG = {
    "host": "mongodb+srv://fonty.hquocfa.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority",
    "tlsCertificateKeyFile": "auth/X509-cert-6658948524510096383.pem",
    "tls": True,
}

MONGO_DATABASE = "fonty"
MONGO_GLYPHS = "glyphs"
MONGO_FONTS = "fonts"

# *----------------------------------------------------------------------------* Client Class


class Client:
    """Base class providing MongoDB client for various data operations."""

    def __init__(self):
        self.client = MongoClient(**MONGO_CONFIG)
        self.glyphs = self.client[MONGO_DATABASE][MONGO_GLYPHS]
        self.fonts = self.client[MONGO_DATABASE][MONGO_FONTS]
