# ⚠️ Deprecated — merged into agentic-memory

agentic-embed has been merged into [agentic-memory](https://github.com/momomo-agent/agentic-memory).

Use `createKnowledgeStore()` from agentic-memory instead:

```js
const { createKnowledgeStore } = require('agentic-memory')
const store = createKnowledgeStore()
await store.learn('id', 'text')
const results = await store.recall('query')
```

See [agentic-memory docs](https://momomo-agent.github.io/agentic-memory/) for details.
