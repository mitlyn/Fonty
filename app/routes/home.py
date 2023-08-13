import falcon
import aiofiles


class HomeRoute:
    async def on_get(self, I, O):
        O.status = falcon.HTTP_200
        O.content_type = 'text/html'
        O.stream = await aiofiles.open('index.html', 'rb')

    async def on_options(self, I, O):
        O.content_type = falcon.MEDIA_TEXT
        O.status = falcon.HTTP_200
        O.text = "OK"
