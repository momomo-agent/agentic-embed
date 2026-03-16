# ⚡ agentic-embed

Vector search for AI. Text → chunks → embeddings → semantic search.

Zero dependencies. Works with [agentic-lite](https://github.com/momomo-agent/agentic-lite) or standalone.

## Why

RAG (Retrieval-Augmented Generation) is the most common AI pattern beyond chat. But existing tools are heavy — LangChain's vector stores, Pinecone SDKs, LlamaIndex pipelines.

agentic-embed: one file, one `create()`, text in, results out.

## Install

```bash
npm install agentic-embed
```

Or:

```html
<script src="https://unpkg.com/agentic-embed/embed.js"></script>
```

## Quick Start

```js
import { create } from 'agentic-embed'

const store = create({
  provider: 'openai',
  apiKey: process.env.OPENAI_API_KEY,
})

// Add documents (auto-chunked)
await store.add('doc-1', 'Quantum computing uses qubits to perform calculations...')
await store.add('doc-2', 'Neural networks are inspired by biological brains...')
await store.add('doc-3', 'CRISPR allows precise editing of DNA sequences...')

// Search
const results = await store.search('How do quantum computers work?')
// → [{ id: 'doc-1', chunk: '...', score: 0.89, ... }]
```

## With agentic-lite

```js
import { ask } from 'agentic-lite'
import { create } from 'agentic-embed'

const store = create({ apiKey: '...' })
await store.add('docs', longDocumentText)

async function ragChat(question) {
  const context = await store.search(question, { topK: 3 })
  const prompt = `Based on this context:\n${context.map(r => r.chunk).join('\n\n')}\n\nAnswer: ${question}`
  return ask(prompt, { apiKey: '...' })
}
```

## Features

### Auto-chunking

Long documents are automatically split into chunks with configurable size and overlap:

```js
const store = create({
  apiKey: '...',
  chunkOptions: {
    maxChunkSize: 500,  // chars per chunk
    overlap: 50,        // overlap between chunks
    separator: '\n\n',  // split on paragraphs
  }
})
```

### Local mode (no API needed)

TF-IDF-based vectors for testing and offline use:

```js
const store = create({ provider: 'local' })
await store.add('a', 'Cats are domestic animals')
await store.add('b', 'Dogs are loyal pets')
await store.search('pet animals')
```

### Batch add

```js
await store.addMany([
  { id: 'doc-1', text: '...', metadata: { source: 'wiki' } },
  { id: 'doc-2', text: '...', metadata: { source: 'arxiv' } },
])
```

### Filter and threshold

```js
const results = await store.search('quantum', {
  topK: 5,
  threshold: 0.7,                    // minimum similarity
  filter: (e) => e.metadata.source === 'arxiv',
  dedupe: true,                      // one result per document
})
```

### Any OpenAI-compatible endpoint

```js
const store = create({
  provider: 'openai',
  baseUrl: 'https://my-proxy.com/v1',
  apiKey: '...',
  model: 'text-embedding-3-small',
})
```

## API

### `create(options?)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | `string` | `'openai'` | `'openai'` or `'local'` |
| `apiKey` | `string` | `null` | API key for embedding provider |
| `baseUrl` | `string` | `null` | Custom endpoint URL |
| `model` | `string` | `null` | Embedding model name |
| `chunkOptions` | `object` | `{}` | `{ maxChunkSize, overlap, separator }` |
| `batchSize` | `number` | `100` | Max texts per API call |

### Instance Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `store.add(id, text, metadata?)` | `Promise` | Add document (auto-chunked) |
| `store.addMany(docs)` | `Promise` | Batch add `[{ id, text, metadata? }]` |
| `store.search(query, opts?)` | `Promise<Result[]>` | Semantic search |
| `store.remove(id)` | `this` | Remove document by ID |
| `store.ids()` | `string[]` | List document IDs |
| `store.size` | `number` | Document count |
| `store.chunkCount` | `number` | Total chunks |
| `store.clear()` | `this` | Remove all |
| `store.export()` | `object` | Serialize state |
| `store.import(data)` | `this` | Restore state |

### Search Result

```js
{
  id: 'doc-1',
  chunk: 'The relevant text chunk...',
  score: 0.892,
  chunkIndex: 2,
  totalChunks: 5,
  metadata: { source: 'wiki' }
}
```

### Utilities

```js
import { chunkText, cosineSimilarity, localEmbed } from 'agentic-embed'

chunkText('Long text...', { maxChunkSize: 500 })
cosineSimilarity([0.1, 0.2], [0.3, 0.4])
localEmbed(['text 1', 'text 2'])  // TF-IDF vectors, no API
```

## Size

~12KB raw, ~3KB gzip. Zero dependencies.

## License

MIT
