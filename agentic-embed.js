/**
 * agentic-embed — Vector search for AI
 * Zero dependencies. Text → chunks → embeddings → search.
 *
 * Usage:
 *   const store = AgenticEmbed.create({ apiKey: '...' })
 *   await store.add('doc-1', 'Quantum computing uses qubits...')
 *   await store.add('doc-2', 'Neural networks are inspired by...')
 *   const results = await store.search('How do qubits work?', { topK: 3 })
 */
;(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory()
  else if (typeof define === 'function' && define.amd) define(factory)
  else root.AgenticEmbed = factory()
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict'

  // ── Text chunking ────────────────────────────────────────────────

  function chunkText(text, options = {}) {
    const {
      maxChunkSize = 500,    // chars per chunk
      overlap = 50,           // overlap between chunks
      separator = null,       // custom separator regex/string
    } = options

    if (!text || text.length <= maxChunkSize) {
      return [text]
    }

    const chunks = []

    if (separator) {
      // Split by separator first, then merge into chunks
      const parts = typeof separator === 'string'
        ? text.split(separator)
        : text.split(separator)

      let current = ''
      for (const part of parts) {
        if (current.length + part.length > maxChunkSize && current.length > 0) {
          chunks.push(current.trim())
          // Keep overlap from end of current chunk
          current = current.slice(-overlap) + part
        } else {
          current += (current ? (typeof separator === 'string' ? separator : '\n') : '') + part
        }
      }
      if (current.trim()) chunks.push(current.trim())
      return chunks
    }

    // Default: split on paragraph/sentence boundaries
    // First try paragraphs
    const paragraphs = text.split(/\n\n+/)

    let current = ''
    for (const para of paragraphs) {
      if (current.length + para.length + 2 > maxChunkSize && current.length > 0) {
        chunks.push(current.trim())
        current = current.slice(-overlap)
      }
      current += (current ? '\n\n' : '') + para
    }
    if (current.trim()) chunks.push(current.trim())

    // If any chunk is still too large, split by sentences
    const result = []
    for (const chunk of chunks) {
      if (chunk.length <= maxChunkSize) {
        result.push(chunk)
        continue
      }
      // Split by sentence endings
      const sentences = chunk.match(/[^.!?]+[.!?]+\s*|[^.!?]+$/g) || [chunk]
      let cur = ''
      for (const sent of sentences) {
        if (cur.length + sent.length > maxChunkSize && cur.length > 0) {
          result.push(cur.trim())
          cur = cur.slice(-overlap)
        }
        cur += sent
      }
      if (cur.trim()) result.push(cur.trim())
    }

    return result
  }

  // ── Cosine similarity ────────────────────────────────────────────

  function cosineSimilarity(a, b) {
    let dot = 0, magA = 0, magB = 0
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i]
      magA += a[i] * a[i]
      magB += b[i] * b[i]
    }
    magA = Math.sqrt(magA)
    magB = Math.sqrt(magB)
    if (magA === 0 || magB === 0) return 0
    return dot / (magA * magB)
  }

  // ── Embedding providers ──────────────────────────────────────────

  async function fetchWithRetry(url, options, retries = 2) {
    for (let i = 0; i <= retries; i++) {
      try {
        const res = await fetch(url, options)
        if (res.status === 429) {
          const wait = parseInt(res.headers.get('retry-after') || '2') * 1000
          await new Promise(r => setTimeout(r, wait))
          continue
        }
        if (!res.ok) {
          const body = await res.text().catch(() => '')
          throw new Error(`Embedding API error ${res.status}: ${body.slice(0, 200)}`)
        }
        return res
      } catch (err) {
        if (i === retries) throw err
        await new Promise(r => setTimeout(r, 1000 * (i + 1)))
      }
    }
  }

  const providers = {
    async openai(texts, config) {
      const baseUrl = config.baseUrl || 'https://api.openai.com/v1'
      const model = config.model || 'text-embedding-3-small'
      const res = await fetchWithRetry(`${baseUrl}/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.apiKey}`,
        },
        body: JSON.stringify({ input: texts, model }),
      })
      const data = await res.json()
      return data.data.map(d => d.embedding)
    },

    // Voyager, Cohere, etc. use OpenAI-compatible endpoints
    async compatible(texts, config) {
      return providers.openai(texts, config)
    },
  }

  // ── Local fallback: TF-IDF-like vectors ──────────────────────────
  // No API needed. Obviously worse than real embeddings, but useful
  // for testing and offline use.

  function localEmbed(texts, vocabSize = 512) {
    // Build vocabulary from all texts
    const vocab = new Map()
    const allTokens = texts.map(t => tokenize(t))

    for (const tokens of allTokens) {
      for (const token of tokens) {
        if (!vocab.has(token)) vocab.set(token, vocab.size % vocabSize)
      }
    }

    // TF-IDF-ish vectors
    const df = new Float32Array(vocabSize) // document frequency
    for (const tokens of allTokens) {
      const seen = new Set()
      for (const token of tokens) {
        const idx = vocab.get(token) ?? (hashStr(token) % vocabSize)
        if (!seen.has(idx)) { df[idx]++; seen.add(idx) }
      }
    }

    return allTokens.map(tokens => {
      const vec = new Float32Array(vocabSize)
      const tf = new Map()
      for (const token of tokens) {
        const idx = vocab.get(token) ?? (hashStr(token) % vocabSize)
        tf.set(idx, (tf.get(idx) || 0) + 1)
      }
      for (const [idx, count] of tf) {
        const idf = Math.log((texts.length + 1) / (df[idx] + 1)) + 1
        vec[idx] = (count / tokens.length) * idf
      }
      // L2 normalize
      let mag = 0
      for (let i = 0; i < vec.length; i++) mag += vec[i] * vec[i]
      mag = Math.sqrt(mag)
      if (mag > 0) for (let i = 0; i < vec.length; i++) vec[i] /= mag
      return Array.from(vec)
    })
  }

  function tokenize(text) {
    return text.toLowerCase()
      .replace(/[^\w\s\u4e00-\u9fff]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 1)
  }

  function hashStr(s) {
    let h = 0
    for (let i = 0; i < s.length; i++) {
      h = ((h << 5) - h + s.charCodeAt(i)) | 0
    }
    return Math.abs(h)
  }

  // ── Core: create store ───────────────────────────────────────────

  function create(options = {}) {
    const {
      provider = 'openai',  // 'openai' | 'local'
      apiKey = null,
      baseUrl = null,
      model = null,
      chunkOptions = {},      // passed to chunkText
      batchSize = 100,        // max texts per embedding API call
    } = options

    const config = { apiKey, baseUrl, model }

    // Store: array of { id, text, chunk, chunkIndex, embedding, metadata }
    let entries = []
    // Vocabulary for local embeddings (rebuilt on each embed call)
    let needsReembed = false

    async function embed(texts) {
      if (provider === 'local') {
        return localEmbed(texts)
      }
      const embedFn = providers[provider] || providers.compatible
      // Batch
      const all = []
      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize)
        const embeddings = await embedFn(batch, config)
        all.push(...embeddings)
      }
      return all
    }

    return {
      /** Add a document (auto-chunked) */
      async add(id, text, metadata = {}) {
        const chunks = chunkText(text, chunkOptions)
        const embeddings = await embed(chunks)

        for (let i = 0; i < chunks.length; i++) {
          entries.push({
            id,
            text,
            chunk: chunks[i],
            chunkIndex: i,
            totalChunks: chunks.length,
            embedding: embeddings[i],
            metadata,
          })
        }

        if (provider === 'local') needsReembed = true
        return this
      },

      /** Add multiple documents at once (more efficient) */
      async addMany(docs) {
        // docs: [{ id, text, metadata? }]
        const allChunks = []
        const chunkMap = [] // track which doc each chunk belongs to

        for (const doc of docs) {
          const chunks = chunkText(doc.text, chunkOptions)
          for (let i = 0; i < chunks.length; i++) {
            allChunks.push(chunks[i])
            chunkMap.push({ ...doc, chunk: chunks[i], chunkIndex: i, totalChunks: chunks.length })
          }
        }

        const embeddings = await embed(allChunks)

        for (let i = 0; i < chunkMap.length; i++) {
          const c = chunkMap[i]
          entries.push({
            id: c.id,
            text: c.text,
            chunk: c.chunk,
            chunkIndex: c.chunkIndex,
            totalChunks: c.totalChunks,
            embedding: embeddings[i],
            metadata: c.metadata || {},
          })
        }

        if (provider === 'local') needsReembed = true
        return this
      },

      /** Search for similar content */
      async search(query, searchOptions = {}) {
        const {
          topK = 5,
          threshold = 0,       // minimum similarity score
          filter = null,       // (entry) => boolean
          dedupe = true,       // deduplicate by document ID
        } = searchOptions

        if (entries.length === 0) return []

        // For local embeddings, re-embed everything when corpus changed
        // (vocabulary shifts affect all vectors)
        if (provider === 'local' && needsReembed) {
          const allTexts = entries.map(e => e.chunk)
          allTexts.push(query) // include query in vocab
          const allEmbeddings = localEmbed(allTexts)
          for (let i = 0; i < entries.length; i++) {
            entries[i].embedding = allEmbeddings[i]
          }
          const queryEmbedding = allEmbeddings[allEmbeddings.length - 1]
          needsReembed = false
          return rankResults(queryEmbedding, { topK, threshold, filter, dedupe })
        }

        const [queryEmbedding] = await embed([query])
        return rankResults(queryEmbedding, { topK, threshold, filter, dedupe })
      },

      /** Remove documents by ID */
      remove(id) {
        entries = entries.filter(e => e.id !== id)
        if (provider === 'local') needsReembed = true
        return this
      },

      /** List all document IDs */
      ids() {
        return [...new Set(entries.map(e => e.id))]
      },

      /** Get document count */
      get size() {
        return new Set(entries.map(e => e.id)).size
      },

      /** Get chunk count */
      get chunkCount() {
        return entries.length
      },

      /** Clear all entries */
      clear() {
        entries = []
        return this
      },

      /** Export store state */
      export() {
        return {
          entries: entries.map(e => ({ ...e })),
          provider,
          model: config.model,
        }
      },

      /** Import store state */
      import(data) {
        entries = (data.entries || []).map(e => ({ ...e }))
        return this
      },
    }

    function rankResults(queryEmbedding, { topK, threshold, filter, dedupe }) {
      let scored = entries.map(entry => ({
        ...entry,
        score: cosineSimilarity(queryEmbedding, entry.embedding),
      }))

      // Filter
      if (filter) scored = scored.filter(filter)
      if (threshold > 0) scored = scored.filter(s => s.score >= threshold)

      // Sort by score
      scored.sort((a, b) => b.score - a.score)

      // Dedupe by document ID (keep highest scoring chunk)
      if (dedupe) {
        const seen = new Set()
        scored = scored.filter(s => {
          if (seen.has(s.id)) return false
          seen.add(s.id)
          return true
        })
      }

      return scored.slice(0, topK).map(s => ({
        id: s.id,
        chunk: s.chunk,
        score: Math.round(s.score * 1000) / 1000,
        chunkIndex: s.chunkIndex,
        totalChunks: s.totalChunks,
        metadata: s.metadata,
      }))
    }
  }

  return {
    create,
    chunkText,
    cosineSimilarity,
    localEmbed,
  }
})
