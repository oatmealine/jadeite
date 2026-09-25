import asyncio
from typing import final

import discord


EDIT_RATELIMIT = 1.0
MAX_LEN = 2000

class MessageTooLong(Exception):
    pass

@final
class LiveMessageWriter:
    def __init__(
        self,
        channel: discord.abc.Messageable,
        existing_message: discord.Message | None = None,

        writing_view: discord.ui.View | None = None,
        complete_view: discord.ui.View | None = None,
    ):
        self.channel = channel
        self.message: discord.Message | None = existing_message
        self._first_message_write = True
        self._writing_view: discord.ui.View | None = writing_view
        self._complete_view: discord.ui.View | None = complete_view
        self.content = ''
        self._dirty = asyncio.Event()
        self.done = False
        self.complete_message = False
        self._task: asyncio.Task[None] | None = None

    def append(self, text: str) -> None:
        if len(self.content) + len(text) > MAX_LEN:
            raise MessageTooLong
        self.content += text
        self._dirty.set()

    async def close(self, complete_message: bool = False) -> None:
        self.done = True
        if complete_message:
            self.complete_message = True
        self._dirty.set() # make sure it removes the trailing █
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        await asyncio.sleep(EDIT_RATELIMIT / 2)
        while True:
            await self._dirty.wait()
            self._dirty.clear()  # clear *before* flushing so appends during the flush re-arm it
            await self._flush()
            if self.done and not self._dirty.is_set():
                return
            await asyncio.sleep(EDIT_RATELIMIT)

    async def _flush(self) -> None:
        content = self.content
        if len(content.strip()) == 0:
            content = '_ _'
        if not self.done and not self.complete_message:
            content += '█'
        if len(content) > MAX_LEN:
            content = content[:MAX_LEN]
        if self.done and not self.complete_message:
            content = content[:-1] + '⋯'

        if self.message is None:
            if self._writing_view is None:
                self.message = await self.channel.send(content)
            else:
                view = self._complete_view if self.done else self._writing_view
                self.message = await self.channel.send(content, view=view)
        else:
            if self.done:
                await self.message.edit(content=content, view=self._complete_view)
            elif self._first_message_write and self._writing_view is not None:
                await self.message.edit(content=content, view=self._writing_view)
            else:
                await self.message.edit(content=content)

        self._first_message_write = False

    async def __aenter__(self):
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc):
        await self.close()
