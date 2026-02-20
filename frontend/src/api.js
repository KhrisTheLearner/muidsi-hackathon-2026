// AgriFlow API client — proxied through Vite dev server to http://localhost:8000

const call = (path, opts = {}) =>
  fetch(path, opts).then(r => r.ok ? r.json() : Promise.reject(r))

export const queryAgent = (query) =>
  call('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })

/**
 * Stream a query via Server-Sent Events for real-time progress.
 * @param {string} query
 * @param {(event: {type:string, [key:string]:any}) => void} onEvent
 * @returns {Promise<void>}
 */
export async function queryAgentStream(query, onEvent) {
  const res = await fetch('/api/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // Parse SSE lines
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete line
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const evt = JSON.parse(line.slice(6))
          onEvent(evt)
        } catch { /* skip malformed */ }
      }
    }
  }
}

export const getHealth   = () => call('/api/health')
export const getExamples = () => call('/api/examples')
export const getCharts   = () => call('/api/charts')
export const getAnalytics = () => call('/api/analytics')
export const getEvalSummary = () => call('/api/eval/summary')

export const planRoute = (origin, destinations) =>
  call('/api/route', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ origin, destinations }),
  })
