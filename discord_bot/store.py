from pathlib import Path
from copy import deepcopy
import json

import aiofiles

store = {
    # map of channel/guild id -> str
    'mode_overrides': {},
    # map of channel/guild id -> str
    'behavior_overrides': {},
}
default_store = deepcopy(store)

PATH = Path.cwd() / 'store.json'

async def load_store():
    data = None

    try:
        async with aiofiles.open(PATH, mode = 'r') as file:
            data = json.loads(await file.read())
    except FileNotFoundError:
        print('no store exists right now')
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f'error loading data: {e}; ignoring')

    if data is None:
        store.update(default_store)
        return

    store.update(data)

async def save_store():
    async with aiofiles.open(PATH, mode = 'w') as file:
        await file.write(json.dumps(store))
