from typing import override

from dotenv import load_dotenv

_ = load_dotenv()

import asyncio
import os

import discord

from llm import llm, get_model_name
from store import load_store

class Jadeite(discord.Bot):
    async def on_ready(self):
        print(f'logged in as {self.user}; ready')
        await self.change_presence(activity=discord.Game(get_model_name()))

    @override
    async def close(self):
        print('bye bye')
        await llm.close()
        await super().close()

intents = discord.Intents.default()
intents.message_content = True

client = Jadeite(intents=intents)

cogs = client.load_extensions('cogs')

loop = asyncio.get_event_loop()

print('loading store...')
loop.run_until_complete(load_store())

print('logging in...')
client.run(os.getenv('DISCORD_TOKEN'))
