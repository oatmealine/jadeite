import os
import asyncio

from openai import AsyncOpenAI


API_BASE = os.getenv('LLAMA_ENDPOINT') or 'http://127.0.0.1:11037/v1'
API_KEY = os.getenv('LLAMA_API_KEY') or 'none'

model_name = '' # default to empty and hope it's not a serve server

def get_model_name():
    return model_name

llm = AsyncOpenAI(
    base_url = API_BASE,
    api_key = API_KEY,
)

async def fetch_model_name():
    global model_name
    model_list = await llm.models.list()
    async for model in model_list:
        name = model.id
        if name.endswith('.gguf'):
            name = name[:-5]
        model_name = name.split('/').pop()
        break

loop = asyncio.get_event_loop()
loop.create_task(fetch_model_name())
