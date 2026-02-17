// AgriFlow API client — proxied through Vite dev server to http://localhost:8000

const call = (path, opts = {}) =>
  fetch(path, opts).then(r => r.ok ? r.json() : Promise.reject(r))

export const queryAgent = (query) =>
  call('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })

export const getHealth   = () => call('/api/health')
export const getExamples = () => call('/api/examples')
export const getCharts   = () => call('/api/charts')
export const getAnalytics = () => call('/api/analytics')
export const getEvalSummary = () => call('/api/eval/summary')
