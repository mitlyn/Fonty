import falcon
import falcon.asgi

from fonts.clients import RemoteClient, LocalClient
from fonts.manager import FontManager

# *----------------------------------------------------------------------------* App & Middleware

app = falcon.asgi.App(
    cors_enable=True,
    middleware=[
        falcon.CORSMiddleware(
            allow_origins="*",
            allow_credentials="*",
        ),
    ],
)

# *----------------------------------------------------------------------------* Resources

class MainRoute:
    async def on_options(self, I, O):
        O.content_type = falcon.MEDIA_TEXT
        O.status = falcon.HTTP_200
        O.text = "OK"


class AddRoute:
    def __init__(self, manager: FontManager):
        self.manager = manager

    async def on_post(self, I, O):
        data = await I.media
        urls = []

        print(data)

        if "url" in data:
            urls.append(data["url"])
        elif "urls" in data:
            urls.extend(data["urls"])
        else:
            raise Exception("No urls provided.")

        ids = manager.addFonts(*urls)

        O.text = f"{len(ids.inserted_ids)} processed fonts."
        O.content_type = falcon.MEDIA_TEXT
        O.status = falcon.HTTP_201


class RenderRoute:
    def __init__(self, manager: FontManager):
        self.manager = manager

    async def on_get(self, I, O):
        ids = manager.renderAll()

        O.text = f"{len(ids.inserted_ids)} rendered fonts."
        O.content_type = falcon.MEDIA_TEXT
        O.status = falcon.HTTP_201


class DropRoute:
    def __init__(self, manager: FontManager):
        self.manager = manager

    async def on_get(self, I, O):
        manager.dropFonts()
        manager.dropGlyphs()

        O.text = "All data has been dropped."
        O.content_type = falcon.MEDIA_TEXT
        O.status = falcon.HTTP_201

# *----------------------------------------------------------------------------* Error Handling

async def ErrorHandler(I, O, ex, params):
    O.content_type = falcon.MEDIA_TEXT
    O.text = f"ERROR: {str(ex)}"
    O.status = falcon.HTTP_500

# *----------------------------------------------------------------------------* Services

localDB = LocalClient()
remoteDB = RemoteClient()
manager = FontManager(remoteDB, localDB)

# *----------------------------------------------------------------------------* Routing

app.add_route("/", MainRoute())

app.add_route("/add", AddRoute(manager))
app.add_route("/drop", DropRoute(manager))
app.add_route("/render", RenderRoute(manager))

app.add_error_handler(Exception, ErrorHandler)