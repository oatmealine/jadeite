import asyncio
from enum import Enum
import re

import discord
from openai import AsyncStream
from openai.types.responses import ResponseInputParam, ResponseStreamEvent
from httpx import ReadError

from llm import llm, get_model_name
from live_message import LiveMessageWriter, MessageTooLong
from store import store, save_store

# TODO: add server emotes to prompt maybe? also format them nicer?
# TODO: implement LLM->chat user mentions
# TODO: implement chat->LLM channel mentions
# TODO: implement LLM->chat channel mentions
# TODO: add tools. have fun
SYSTEM_PROMPT = '''
You are Jadeite. When people refer to "jadeite", assume they are referring to or asking you. Chat with relevancy to the conversation.
You are your own person with your own opinions and ideas.
Messages will be prefixed with who wrote it. You should refer to users with what is specified there. **Do not prefix your messages this way.**
You can mention users by putting an @ in front of their name (like "@jade"). Users can also mention you this way.
Users will expect you to perform tasks as an AI assistant. Do not deflect or attempt to schedule those tasks. Don't say "i'll do it later" or "i'm currently busy".
Avoid excessive emoji use.
Otherwise, act like yourself. Feel free to be silly, kind, negative, serious, affectionate, neutral, whatever you like.
'''

MENTION_PATTERN = re.compile(r'<@!?(\d+)>')

class ResponseMode(Enum):
    # write the message out live
    ASSISTANT = 0,
    # send the whole message at once
    MIMIC = 1,

# represents the state of a given conversation
class Conversation:
    def __init__(
        self,
        bot: discord.Bot,
        channel: discord.abc.Messageable
    ):
        self.channel: discord.abc.Messageable = channel
        self.bot: discord.Bot = bot
        self.delete: bool = False

        self._context: ResponseInputParam = []
        self._last_sent_message: discord.Message | None = None
        self._last_sent_message_has_buttons: bool = False
        self._write_stream: AsyncStream[ResponseStreamEvent] | None = None
        self._is_awaiting_response: bool = False

        self.initialize_system_prompt()

    def initialize_system_prompt(self):
        self._context.append({
            'type': 'message',
            'role': 'system',
            'content': [{ 'text': SYSTEM_PROMPT.strip(), 'type': 'input_text' }]
        })
        self._context.append({
            'type': 'message',
            'role': 'assistant',
            'content': [{ 'text': 'okay, understood, i\'m jadeite', 'type': 'input_text' }]
        })

    def delete_convo(self):
        self.delete = True
        if self._last_sent_message and self._last_sent_message_has_buttons:
            _ = asyncio.create_task(self._last_sent_message.edit(view = None))

    async def _mention_to_user(self, match: re.Match[str]):
        try:
            id = int(match.group(1))
        except ValueError:
            return match.group(0)

        user = await self.bot.get_or_fetch(discord.User, id)
        if user is None:
            return match.group(0)

        return user.display_name

    async def add_message_to_context(self, message: discord.Message):
        content = message.content
        is_me = message.author.id == self.bot.user.id

        for match in MENTION_PATTERN.finditer(content):
            replaced = await self._mention_to_user(match)
            # this is sooooo gonna break
            content = content.replace(match.group(0), replaced)

        if not is_me:
            content = f'{message.author.global_name or message.author.name}: ' + content

        self._context.append({
            'type': 'message',
            'role': 'assistant' if is_me else 'user',
            'content': [{ 'text': content, 'type': 'input_text' }]
        })

    async def respond(
        self,
        message: discord.Message | None = None,
        mode: ResponseMode = ResponseMode.ASSISTANT
    ):
        stream = await llm.responses.create(
            model = get_model_name(),
            input = self._context,
            stream = True,
        )
        self._write_stream = stream

        self._last_sent_message_has_buttons = mode == ResponseMode.ASSISTANT

        if mode == ResponseMode.ASSISTANT:
            if message is None:
                await self.channel.trigger_typing()

            async with LiveMessageWriter(
                self.channel,
                existing_message = message,
                writing_view=WritingView(),
                complete_view=RespondedView(),
            ) as writer:
                try:
                    async for event in stream:
                        if event.type == 'response.output_text.delta':
                            writer.append(event.delta)
                        elif event.type == 'response.output_text.done':
                            await writer.close(complete_message = True)
                            break
                except (MessageTooLong, ReadError):
                    # abrupt ending
                    await stream.close()
                    await writer.close()

            self._last_sent_message = writer.message

            stored_content = writer.content
            complete_message = writer.complete_message
        else:
            content = ""
            complete_message = False

            async with self.channel.typing():
                try:
                    async for event in stream:
                        if event.type == 'response.output_text.delta':
                            content += event.delta
                            if len(content) > 2000:
                                content = content[:2000]
                                await stream.close()
                                break
                        elif event.type == 'response.output_text.done':
                            complete_message = True
                            break
                except ReadError:
                    # abrupt ending
                    await stream.close()

            self._last_sent_message = await self.channel.send(content)
            stored_content = content

        self._write_stream = None
        self._is_awaiting_response = True

        if not complete_message:
            stored_content += "[truncated]"

        self._context.append({
            'type': 'message',
            'role': 'assistant',
            'content': [{ 'text': stored_content, 'type': 'input_text' }]
        })

    async def reply_to_message(self, message: discord.Message, mode: ResponseMode):
        self._is_awaiting_response = False

        await self.add_message_to_context(message)

        await self.respond(None, mode)

    async def retry_last_reply(self):
        if not self._is_awaiting_response:
            return

        self._is_awaiting_response = False

        _ = self._context.pop() # remove current reply

        await self.respond(self._last_sent_message, ResponseMode.ASSISTANT)

    def start_write(self, stream: AsyncStream[ResponseStreamEvent]):
        self._write_stream = stream

    def is_writing(self):
        return self._write_stream is not None

    async def stop_response(self):
        if self._write_stream is not None:
            await self._write_stream.close()

    async def on_message(self, message: discord.Message, mode: ResponseMode):
        if self.is_writing():
            await self.stop_response()
        elif self._is_awaiting_response:
            self._is_awaiting_response = False
            if self._last_sent_message and self._last_sent_message_has_buttons:
                _ = asyncio.create_task(self._last_sent_message.edit(view = None))

        await self.reply_to_message(message, mode)


conversations: dict[int, Conversation] = {}

def get_conversation(channel_id: int) -> Conversation | None:
    if channel_id in conversations:
        convo = conversations[channel_id]
        if convo.delete:
            del_conversation(channel_id)
            return None
        return convo
    return None

def del_conversation(channel_id: int):
    conversations[channel_id].delete_convo()
    del conversations[channel_id]

async def get_or_make_conversation(bot: discord.Bot, channel: discord.abc.Messageable):
    existing_convo = get_conversation(channel.id)
    if existing_convo is not None:
        return existing_convo
    convo = Conversation(bot, channel)
    conversations[channel.id] = convo
    return convo

# the buttons seen when a message is being written
class WritingView(discord.ui.View):
    @discord.ui.button(label='︎⏹', style=discord.ButtonStyle.danger)
    async def button_callback(self, _button, interaction: discord.Interaction):
        _ = asyncio.create_task(interaction.response.defer())
        convo = get_conversation(interaction.channel_id)
        if convo is None:
            return
        await convo.stop_response()

# the buttons seen after a response is written, until it's no longer the last
# message
class RespondedView(discord.ui.View):
    @discord.ui.button(label='↻', style=discord.ButtonStyle.primary)
    async def button_callback(self, _button, interaction: discord.Interaction):
        _ = asyncio.create_task(interaction.response.defer())
        convo = get_conversation(interaction.channel_id)
        if convo is None:
            return
        await convo.retry_last_reply()

class ChannelBehavior(Enum):
    # don't respond
    IGNORE = 0,
    # only respond to mentions, fetching messages above
    #ONLY_IF_INVOKED = 1, # scrapped. not very useful
    # only respond to mentions
    IF_MENTIONED = 1,
    # always respond to everything
    ANNOYING = 2,

class Responses(discord.Cog):
    def __init__(self, bot: discord.Bot):
        self.bot: discord.Bot = bot
        pass

    @staticmethod
    def _get_mode(channel_id: int) -> ResponseMode:
        mode_str = store['mode_overrides'].get(str(channel_id), '')
        return getattr(ResponseMode, mode_str.upper(), ResponseMode.ASSISTANT)

    @staticmethod
    def _get_behavior(channel_id: int, is_dm: bool = False) -> ChannelBehavior:
        behavior_str = store['behavior_overrides'].get(str(channel_id), '')
        return getattr(ChannelBehavior, behavior_str.upper(), ChannelBehavior.ANNOYING if is_dm else ChannelBehavior.IF_MENTIONED)

    @discord.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.id == self.bot.user.id:
            return

        behavior = self._get_behavior(message.channel.id, message.channel.type == discord.ChannelType.private)
        mode = self._get_mode(message.channel.id)

        if behavior == ChannelBehavior.IGNORE:
            return

        replying_to = message.reference and message.reference.cached_message
        mentioning_me = (self.bot.user in message.mentions) or (replying_to is not None and replying_to.author.id == self.bot.user.id)
        should_reply = mentioning_me or behavior == ChannelBehavior.ANNOYING

        if not should_reply:
            return

        convo = await get_or_make_conversation(self.bot, message.channel)

        await convo.on_message(message, mode)

    @discord.slash_command(description='clears the current conversation\'s context')
    async def clear(self, ctx: discord.ApplicationContext):
        del_conversation(ctx.channel_id)
        await ctx.respond('context cleared!')

    @discord.slash_command(description='regenerates the last message')
    async def regen(self, ctx: discord.ApplicationContext):
        convo = get_conversation(ctx.channel_id)
        if convo is None:
            await ctx.respond('no current conversation here!', ephemeral = True)
            return
        _ = asyncio.create_task(ctx.respond('OK', ephemeral = True))
        await convo.retry_last_reply()

    @discord.slash_command(description='change the current mode')
    @discord.default_permissions(manage_messages=True)
    async def mode(
        self,
        ctx: discord.ApplicationContext,
        mode: discord.Option(str, choices = ['assistant', 'mimic']),
    ):
        store['mode_overrides'][str(ctx.channel_id)] = mode
        await ctx.respond(f'changed mode to {mode}')
        await save_store()

    @discord.slash_command(description='change the current behavior')
    @discord.default_permissions(manage_messages=True)
    async def behavior(
        self,
        ctx: discord.ApplicationContext,
        behavior: discord.Option(str, choices = ['ignore', 'if mentioned', 'annoying']),
    ):
        store['behavior_overrides'][str(ctx.channel_id)] = behavior
        await ctx.respond(f'changed behavior to {behavior}')
        await save_store()

def setup(bot: discord.Bot):
    bot.add_cog(Responses(bot))
