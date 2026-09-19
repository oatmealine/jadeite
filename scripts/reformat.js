// reformats DiscordChatExporter CSVs to ChatML conversation JSONLs

import * as csv from '@fast-csv/parse';
import { parseArgs } from 'node:util';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    // your own ID; the "assistant"
    ['self-id']: { type: 'string' },
    // date before which messages should get discarded. accepts anything js will parse
    ['cutoff-date']: { type: 'string', default: '1/1/1970' },
    // time after which conversations should be considered seperate and unrelated, in seconds
    ['conversation-interval']: { type: 'string', default: '1200' }
  }
});

/** @typedef {{ authorId: string, author: string, timestamp: string, content: string, attachments: string, reactions: string }} Message */

if (!values['self-id'] || positionals.length === 0) {
  process.stderr.write('usage: reformat.js --self-id [ID] [FILES] > out.jsonl\n');
  process.exit(1);
}

const SELF_ID = values['self-id'];
const CUTOFF_THRESHOLD = Date.parse(values['cutoff-date']);
const CONVERSATION_INTERVAL = parseInt(values['conversation-interval']) * 1_000;

let totalConvos = 0;

for (const file of positionals) {
  const writeProgress = (convos) => {
    process.stderr.cursorTo(0);
    const prefix = convos.toString().padEnd(4, ' ');
    process.stderr.write(prefix);
    const filename = file.split('/').pop().replace('.csv', '');
    process.stderr.write(filename.slice(Math.max(filename.length - (process.stderr.columns - prefix.length), 0)));
  };
  writeProgress(0);

  //const totalSize = (await fsp.stat(path)).size;

  /** @type Message[] */
  const buffer = [];
  /** @type {number?} */
  let prevTimestamp = null;
  let convos = 0;

  const flushBuffer = () => {
    if (buffer.length === 0) return;

    const messages = buffer.splice(0, buffer.length);

    let conversation = [];
    let lastAuthor = null;
    let currentContent = '';

    for (const message of messages) {
      // no convos w/ attachments
      if (message.attachments !== '') return;
      // no messages w/ links [TODO: filter them out maybe?]
      if (message.content.includes('https://') || message.content.includes('http://')) continue;
      // no messages w/o content
      if (message.content === '') continue;


      if (lastAuthor && lastAuthor !== message.authorId) {
        conversation.push({
          role: lastAuthor === SELF_ID ? 'assistant' : 'user',
          content: currentContent,
        });
        currentContent = '';
      }

      lastAuthor = message.authorId;

      if (currentContent.length > 0) {
        currentContent += '\n';
      }
      currentContent += message.content;
    }
    if (currentContent !== '') {
      conversation.push({
        role: lastAuthor === SELF_ID ? 'assistant' : 'user',
        content: currentContent,
      });
      currentContent = '';
    }

    // make conversations always start with the user
    while (conversation.length > 0 && conversation[0].role === 'assistant') {
      conversation.splice(0, 1);
    }
    // and always end with the assistant
    while (conversation.length > 0 && conversation[conversation.length - 1].role === 'user') {
      conversation.pop();
    }

    if (
      conversation.some(msg => msg.role === 'user') &&
      conversation.some(msg => msg.role === 'assistant')
    ) {
      convos++;
      process.stdout.write(JSON.stringify({
        messages: conversation
      }) + '\n');
      writeProgress(convos);
    }
  };

  await new Promise(resolve => {
    csv.parseFile(file, {
      headers: [ 'authorId', 'author', 'timestamp', 'content', 'attachments', 'reactions' ],
      renameHeaders: true
    })
      .on('error', error => process.stderr.write(error + '\n'))
      .on('data', row => {
        const timestamp = Date.parse(row.timestamp);
        // no convos too old
        if (timestamp < CUTOFF_THRESHOLD) return;

        buffer.push(row);

        if (prevTimestamp && timestamp - prevTimestamp > CONVERSATION_INTERVAL) {
          flushBuffer();
        }
        prevTimestamp = timestamp;
      })
      .on('end', () => {
        flushBuffer();
        resolve();
      });
  });

  process.stderr.write('\n');
  totalConvos += convos;
}

process.stderr.write(`\n${totalConvos} total convos exported\n`);
