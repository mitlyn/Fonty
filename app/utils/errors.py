import falcon


async def ErrorHandler(I, O, ex, params):
    O.content_type = falcon.MEDIA_TEXT
    O.text = f"ERROR: {str(ex)}"
    O.status = falcon.HTTP_500
