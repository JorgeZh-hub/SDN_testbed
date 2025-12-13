import asyncio
from aiocoap import *
from aiocoap import resource
import socket

class SensorResource(resource.Resource):
    def __init__(self):
        super().__init__()
        self.last_data = b'{}'

    async def render_put(self, request):
        self.last_data = request.payload
        #print("Recibido:", self.last_data.decode())
        # print("Recibido:", self.last_data.decode())
        return Message(code=CHANGED, payload=b"OK")

    async def render_get(self, request):
        return Message(code=CONTENT, payload=self.last_data)

root = resource.Site()
root.add_resource(['sensor', 'datos'], SensorResource())

async def main():
    bind_address = ('0.0.0.0', 5683)
    await Context.create_server_context(root, bind=bind_address)
    await asyncio.get_event_loop().create_future()

asyncio.run(main())
