import argparse
from datetime import datetime
import sys
from sys import stderr
import math
import re
import json
from pytz import UTC

from atproto import Client, IdResolver, AtUri
from atproto_client.models.app.bsky.feed.post import Record as AppBskyFeedPost
from atproto_client.exceptions import BadRequestError, RequestException
from pydantic_core._pydantic_core import ValidationError

# https://stackoverflow.com/a/2979208
def entropy(string: str) -> float:
    'Calculates the Shannon entropy of a string'

    # get probability of chars in string
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]

    # calculate the entropy
    entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])

    return entropy

def match_case(orig: str, repl: str) -> str:
    if orig.isupper():
        return repl.upper()
    if orig[0].isupper():
        return repl.capitalize()
    return repl.lower()

parser = argparse.ArgumentParser(
    description = '''
    takes a Bluesky handle (@oat.zone) or DID scrapes its reply threads into a conversations .jsonl.
    this will dump the resulting .jsonl to STDOUT, you should pipe it to a file
    '''
)

def valid_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f'not a valid date: {s!r}')

parser.add_argument('--handle', help = 'your ATProto handle (@oat.zone)')
parser.add_argument('--did', help = 'your ATProto DID (did:plc:... or did:web:...)')

parser.add_argument('--cutoff-date', type = valid_date, default = datetime.strptime('1970-01-01', '%Y-%m-%d'), help = 'only consider posts after this date (as yyyy-mm-dd)')
parser.add_argument('--redact', help = 'for names (comma-separated). this will redact those names and replace them with some random name instead. highly recommended; just go through your training data sources and think of the names that might be referred to in that data. requires the "faker" python module')
parser.add_argument('--no-double-blocks', action = 'store_true', help = 'when two people that aren\'t you speak in a thread, their posts are not joined in one "user" block. this disables that for models w/ templates that don\'t allow this (inapplicable with --raw)')
parser.add_argument('--raw', action = 'store_true', help = 'gives you your text and your text alone in a raw format, for use in pre-training')

args = parser.parse_args()

if (args.did is None and args.handle is None) or (args.did is not None and args.handle is not None):
    print(f'{parser.prog}: error: either --did or --handle (but not both) is required')
    sys.exit(2)

redact_names = list(
    map(
        lambda name: re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE),
        filter(
            lambda name: len(name),
            map(
                lambda name: name.strip(),
                (args.redact or '').split(',')
            )
        )
    )
)

if redact_names:
    from faker import Faker
    fake = Faker()

resolver = IdResolver()

if args.handle:
    did = resolver.handle.resolve(args.handle)
else:
    did = args.did

did_doc = resolver.did.resolve(did)
did = did_doc.id
handle = did_doc.get_handle()

client = Client(base_url = did_doc.get_pds_endpoint())
slingshot = Client(base_url = 'https://slingshot.microcosm.blue')

cursor = None

# list of seen URIs
seen_posts: list[str] = []

MAX_DEPTH = 20

# rather clumsy recursion argument passing, but whatever
def fetch_thread(uri: str, post: AppBskyFeedPost, depth: int = 0) -> list[tuple[str, AppBskyFeedPost]]:
    posts: list[tuple[str, AppBskyFeedPost]] = []

    posts.append((uri, post))

    if depth > MAX_DEPTH:
        return posts

    if getattr(post, 'reply', None) is not None:
        parent_uri = post.reply.parent.uri
        if parent_uri in seen_posts:
            return posts
        parent_ref = AtUri.from_str(parent_uri)
        stderr.write(('  ' * (depth + 1)) + f'fetching up: {parent_ref}\n')
        try:
            res = slingshot.app.bsky.feed.post.get(parent_ref.hostname, parent_ref.rkey)
        except (RequestException, BadRequestError, ValidationError) as e:
            stderr.write(('  ' * (depth + 1)) + f'ERR: {e}; ignoring\n')
            return posts

        parent_post = res.value

        seen_posts.append(parent_uri)
        posts = fetch_thread(parent_uri, parent_post, depth + 1) + posts

    return posts

while True:
    posts = client.app.bsky.feed.post.list(did, limit=20, cursor=cursor)

    cursor = posts.cursor

    for uri, post in posts.records.items():
        if uri in seen_posts:
            continue

        if datetime.fromisoformat(post.created_at) < UTC.localize(args.cutoff_date):
            stderr.write(f'reached cutoff date\n')
            cursor = None # evil parent loop break method
            break

        stderr.write(f'{uri}\n')

        seen_posts.append(uri)

        thread = fetch_thread(uri, post)

        if not len(thread):
            continue

        conversation: list[dict[str, str]] = []
        last_author = ''

        for (uri, post) in thread:
            ref = AtUri.from_str(uri)
            is_me = ref.hostname == did

            # remove posts w/ no content
            if not post.text:
                continue
            # remove posts w/ links
            if post.facets and any(any(feature.py_type == 'app.bsky.richtext.facet#link' for feature in facet.features) for facet in post.facets):
                continue

            if args.raw:
                if not is_me:
                    continue
                content = post.text
                for pattern in redact_names:
                    fake_name = fake.first_name()
                    content = pattern.sub(
                        lambda match: match_case(match.group(0), fake_name),
                        content
                    )
                print(content)
                continue

            role = 'assistant' if is_me else 'user'

            should_join = len(conversation) and last_author == ref.hostname
            if args.no_double_blocks:
                should_join = len(conversation) and (last_author == did) == is_me

            if should_join:
                conversation[-1]['content'] += '\n' + post.text
            else:
                conversation.append({
                    'role': role,
                    'content': post.text,
                })

            last_author = ref.hostname

        # remove trailing user blocks (useless in training)
        while len(conversation) and conversation[-1]['role'] == 'user':
            _ = conversation.pop()

        # remove empty convos
        if not len(conversation):
            continue

        # exclude convos with no assistant lines
        if not any(message['role'] == 'assistant' for message in conversation):
            continue
        # exclude convos with no user lines
        if not any(message['role'] == 'user' for message in conversation):
            continue

        # redact names (but keep all references to the same name identical)
        for pattern in redact_names:
            fake_name = fake.first_name()
            for message in conversation:
                message['content'] = pattern.sub(
                    lambda match: match_case(match.group(0), fake_name),
                    message['content']
                )

        weight = 1.0

        # downweight really short replies
        lengths = [len(m['content']) for m in conversation if m['role'] == 'assistant']
        avg_length = sum(lengths) / len(lengths)
        if avg_length < 5:
            weight *= (avg_length + 1) / 7

        # downweight exceptionally low entropy
        entropies = [entropy(m['content']) for m in conversation if m['role'] == 'assistant']
        avg_entropy = sum(entropies) / len(entropies)
        if avg_entropy < 0.7:
            weight *= 0.5

        if weight <= 0.0:
            continue

        data: dict[str, object] = { 'messages': conversation }
        if weight != 1.0:
            data['weight'] = weight

        print(json.dumps(data, separators = (',', ':'), ensure_ascii = False))

    if not cursor:
        break
