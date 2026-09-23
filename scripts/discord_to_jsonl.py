import argparse
from datetime import datetime
import csv
import json
from pytz import UTC
import re
import math
from sys import stderr

# https://stackoverflow.com/a/2979208
def entropy(string: str) -> float:
    'Calculates the Shannon entropy of a string'

    # get probability of chars in string
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]

    # calculate the entropy
    entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])

    return entropy

def valid_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f'not a valid date: {s!r}')

parser = argparse.ArgumentParser(
    description = '''
    take DiscordChatExporter csvs and turn them into a conversations .jsonl.
    this will dump the resulting .jsonl to STDOUT, you should pipe it to a file
    '''
)

parser.add_argument('file', nargs='+')
parser.add_argument('--self-id', type = int, required = True, help = 'your own Discord ID; the "assistant"')

parser.add_argument('--cutoff-date', type = valid_date, default = datetime.strptime('1970-01-01', '%Y-%m-%d'), help = 'only consider messages after this date (as yyyy-mm-dd)')
parser.add_argument('--conversation-interval', type = int, default = 1200, help = 'messages seperated by this amount of seconds will be considered seperate conversations (inapplicable with --raw)')
parser.add_argument('--redact', help = 'for names (comma-separated). this will redact those names and replace them with some random name instead. highly recommended; just go through your training data sources and think of the names that might be referred to in that data. requires the "faker" python module')
parser.add_argument('--no-double-blocks', action = 'store_true', help = 'when two users that aren\'t you speak in a channel, their messages are not joined in one "user" block. this disables that for models w/ templates that don\'t allow this (inapplicable with --raw)')
parser.add_argument('--raw', action = 'store_true', help = 'gives you your text and your text alone in a raw format, for use in pre-training')

args = parser.parse_args()

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

def match_case(orig: str, repl: str) -> str:
    if orig.isupper():
        return repl.upper()
    if orig[0].isupper():
        return repl.capitalize()
    return repl.lower()

global_entropy = []

total_convos = 0

for path in args.file:
    stderr.write(f'0    {path.split('/').pop().replace('.csv', '')}')
    with open(path, newline='') as file:
        reader = csv.reader(file)
        _ = next(reader, None) # skip headers

        last_timestamp: datetime | None = None

        message_buffer: list[list[str]] = []
        file_convos = 0

        def pop_messages():
            global file_convos

            if not len(message_buffer):
                return

            messages = message_buffer.copy()
            message_buffer.clear()

            last_author = ''

            conversation: list[dict[str, str]] = []

            for message in messages:
                author_id, _author, _timestamp_str, content, attachments, _reactions = message
                is_me = int(author_id) == args.self_id

                # remove convos w/ attachments involved
                if not args.raw and attachments:
                    return

                # remove messages w/ no content
                if not content:
                    continue
                # remove messages w/ links
                if 'http://' in content or 'https://' in content:
                    continue

                if args.raw:
                    if not is_me:
                        continue
                    file_convos += 1
                    for pattern in redact_names:
                        fake_name = fake.first_name()
                        content = pattern.sub(
                            lambda match: match_case(match.group(0), fake_name),
                            content
                        )
                    print(content)
                    stderr.write(f'\r{file_convos}')
                    stderr.flush()
                    continue

                role = 'assistant' if is_me else 'user'

                should_join = len(conversation) and last_author == author_id
                if args.no_double_blocks:
                    should_join = len(conversation) and (int(last_author) == args.self_id) == is_me

                if should_join:
                    conversation[-1]['content'] += '\n' + content
                else:
                    conversation.append({
                        'role': role,
                        'content': content,
                    })

                last_author = author_id

            # remove trailing user blocks (useless in training)
            while len(conversation) and conversation[-1]['role'] == 'user':
                _ = conversation.pop()

            # remove empty convos
            if not len(conversation):
                return

            # exclude convos with no assistant lines
            if not any(message['role'] == 'assistant' for message in conversation):
                return
            # exclude convos with no user lines
            if not any(message['role'] == 'user' for message in conversation):
                return

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
                return

            data: dict[str, object] = { 'messages': conversation }
            if weight != 1.0:
                data['weight'] = weight

            file_convos += 1

            stderr.write(f'\r{file_convos}')
            stderr.flush()

            print(json.dumps(data, separators = (',', ':'), ensure_ascii = False))

        for message in reader:
            _author_id, _author, timestamp_str, _content, _attachments, _reactions = message
            timestamp = datetime.fromisoformat(timestamp_str)

            # remove messages before the cutoff period
            if timestamp < UTC.localize(args.cutoff_date):
                continue

            message_buffer.append(message)

            if last_timestamp is not None and (timestamp - last_timestamp).total_seconds() >= args.conversation_interval:
                pop_messages()

            last_timestamp = timestamp

        pop_messages()

        stderr.write('\n')
        total_convos += file_convos

stderr.write('\n')
stderr.write(f'{total_convos} total {'messages' if args.raw else 'convos'}\n')
