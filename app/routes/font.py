import aiofiles
import falcon


class FontRoute:
    def __init__(self, config):
        self.config = config
        # other services

    async def on_post(self, I, O):
        # *--------------------------------------------------------------------* Load File
        form = await I.get_media()
        name = None
        path = None

        async for part in form:
            if part.name == 'file':
                name = part.filename
                path = f"{self.config.uuid()}-{name}"
                # path = self.config.storage / f"{self.config.uuid()}-{name}"

                # TODO: process file using BytesIO
                async with aiofiles.open(path, 'wb') as X:
                    await part.stream.pipe(X)

        # *--------------------------------------------------------------------* Process File

        # TODO: preprocess, extend and vectorize the font

        # *--------------------------------------------------------------------* Return File

        O.set_header("Content-Disposition", f"attachment; filename={name}")
        O.stream = await aiofiles.open(path, 'rb')
        O.content_type = 'font/ttf'
        O.status = falcon.HTTP_200
