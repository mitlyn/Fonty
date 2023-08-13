import falcon
import falcon.asgi
from falcon import CORSMiddleware

from services import *
from routes import *
from utils import *

# *----------------------------------------------------------------------------* App & Middleware

app = falcon.asgi.App(
    cors_enable=True,
    middleware=[
        CORSMiddleware(
            allow_origins="*",
            allow_credentials="*",
        ),
    ],
)

# *----------------------------------------------------------------------------* Services

config = ConfigService()
# extender = Extender()
# vectorizer = Vectorizer()

# *----------------------------------------------------------------------------* Routing

app.add_route("/", HomeRoute())
app.add_route("/font", FontRoute(config))
app.add_error_handler(Exception, ErrorHandler)