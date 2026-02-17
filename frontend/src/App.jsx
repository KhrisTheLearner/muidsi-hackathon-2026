/**
 * AgriFlow — Food Supply Chain Intelligence
 * Professional 4-tab React interface connecting to FastAPI backend.
 * Tabs: Query · Dashboard · Data Sources · Map & Alerts
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { queryAgent, getHealth, getExamples, getCharts, getAnalytics, getEvalSummary } from './api'

// ─── Theme ─────────────────────────────────────────────────────────────────
const T = {
  bg:        '#0b0f1a',
  surface:   'rgba(255,255,255,0.03)',
  border:    'rgba(255,255,255,0.07)',
  borderHi:  'rgba(255,255,255,0.12)',
  green:     '#10b981',
  greenDim:  'rgba(16,185,129,0.12)',
  greenBd:   'rgba(16,185,129,0.25)',
  text:      '#e2e8f0',
  sub:       '#94a3b8',
  muted:     '#64748b',
  danger:    '#ef4444',
  warn:      '#f59e0b',
  blue:      '#3b82f6',
  purple:    '#8b5cf6',
  mono:      "'JetBrains Mono', 'Fira Code', monospace",
  sans:      "'DM Sans', -apple-system, sans-serif",
}

// ─── Shared styles ─────────────────────────────────────────────────────────
const card = {
  background: T.surface,
  border: `1px solid ${T.border}`,
  borderRadius: 12,
  padding: 16,
}

// ─── CSS keyframes injected once ───────────────────────────────────────────
const GLOBAL_CSS = `
  @keyframes spin    { to { transform: rotate(360deg) } }
  @keyframes fadeUp  { from { opacity:0; transform:translateY(10px) } to { opacity:1; transform:translateY(0) } }
  @keyframes pulse   { 0%,100% { opacity:.4 } 50% { opacity:1 } }
  button { font-family: inherit; }
  input, textarea { font-family: inherit; }
`

// ─── Icons ──────────────────────────────────────────────────────────────────
const Ic = {
  Leaf: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.9C15.5 4.9 17 3.5 19 2c1 2 2 4.5 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>,
  Send: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>,
  Chat: () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>,
  Grid: () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>,
  Db:   () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
  Map:  () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>,
  Check: () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>,
  Spin: ({ color = T.green }) => <div style={{ width:12, height:12, borderRadius:'50%', border:`2px solid ${color}`, borderTopColor:'transparent', animation:'spin .7s linear infinite' }} />,
}

// ─── Markdown renderer (lightweight, no deps) ──────────────────────────────
function Md({ text }) {
  if (!text) return null
  const lines = text.split('\n')
  const elements = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // Horizontal rule
    if (/^-{3,}$/.test(line.trim())) {
      elements.push(<hr key={i} style={{ border:'none', borderTop:`1px solid ${T.border}`, margin:'12px 0' }} />)
      i++; continue
    }

    // Headings
    const hMatch = line.match(/^(#{1,4})\s+(.+)/)
    if (hMatch) {
      const level = hMatch[1].length
      const sizes = { 1:16, 2:14.5, 3:13, 4:12.5 }
      elements.push(
        <p key={i} style={{ fontSize:sizes[level], fontWeight:700, color:T.text, margin:'14px 0 6px' }}>
          {inlineFormat(hMatch[2])}
        </p>
      )
      i++; continue
    }

    // Unordered list items
    if (/^\s*[-*]\s+/.test(line)) {
      const items = []
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        const indent = lines[i].match(/^(\s*)/)[1].length
        items.push(
          <li key={i} style={{ marginLeft: indent > 1 ? 16 : 0, marginBottom:3 }}>
            {inlineFormat(lines[i].replace(/^\s*[-*]\s+/, ''))}
          </li>
        )
        i++
      }
      elements.push(<ul key={`ul-${i}`} style={{ margin:'6px 0', paddingLeft:20, listStyleType:'disc' }}>{items}</ul>)
      continue
    }

    // Ordered list items
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = []
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(
          <li key={i} style={{ marginBottom:3 }}>
            {inlineFormat(lines[i].replace(/^\s*\d+\.\s+/, ''))}
          </li>
        )
        i++
      }
      elements.push(<ol key={`ol-${i}`} style={{ margin:'6px 0', paddingLeft:20 }}>{items}</ol>)
      continue
    }

    // Empty line = spacing
    if (!line.trim()) {
      elements.push(<div key={i} style={{ height:8 }} />)
      i++; continue
    }

    // Regular paragraph
    elements.push(<p key={i} style={{ margin:'3px 0', lineHeight:1.65 }}>{inlineFormat(line)}</p>)
    i++
  }

  return <div>{elements}</div>
}

function inlineFormat(text) {
  // Split by inline patterns and return mixed text/elements
  const parts = []
  let remaining = text
  let key = 0

  while (remaining) {
    // Code: `...`
    let m = remaining.match(/^(.*?)`([^`]+)`(.*)$/)
    if (m) {
      if (m[1]) parts.push(m[1])
      parts.push(
        <code key={key++} style={{ background:'rgba(255,255,255,.08)', padding:'1px 5px', borderRadius:4, fontSize:'0.9em', fontFamily:T.mono, color:'#6ee7b7' }}>
          {m[2]}
        </code>
      )
      remaining = m[3]; continue
    }
    // Bold: **...**
    m = remaining.match(/^(.*?)\*\*(.+?)\*\*(.*)$/)
    if (m) {
      if (m[1]) parts.push(m[1])
      parts.push(<strong key={key++} style={{ fontWeight:600, color:T.text }}>{m[2]}</strong>)
      remaining = m[3]; continue
    }
    // Italic: *...*
    m = remaining.match(/^(.*?)\*(.+?)\*(.*)$/)
    if (m) {
      if (m[1]) parts.push(m[1])
      parts.push(<em key={key++}>{m[2]}</em>)
      remaining = m[3]; continue
    }
    // No more patterns
    parts.push(remaining)
    break
  }
  return parts.length === 1 ? parts[0] : parts
}

// ─── PlotlyChart ────────────────────────────────────────────────────────────
function PlotlyChart({ spec, height = 280 }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current || !window.Plotly || !spec) return
    const layout = {
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: T.sub, family: T.sans, size: 11 },
      margin: { t: 20, r: 10, b: 40, l: 10 },
      height,
      ...spec.layout,
    }
    window.Plotly.newPlot(ref.current, spec.data || [], layout, {
      displayModeBar: false, responsive: true,
    })
    return () => window.Plotly?.purge(ref.current)
  }, [spec, height])
  if (!spec) return null
  return <div ref={ref} style={{ width: '100%' }} />
}

// ─── ReasoningStep ──────────────────────────────────────────────────────────
const STEP_COLORS = {
  data:    { bg:'rgba(59,130,246,.12)',   bd:'rgba(59,130,246,.3)',   fg:'#60a5fa' },
  sql:     { bg:'rgba(99,102,241,.12)',   bd:'rgba(99,102,241,.3)',   fg:'#818cf8' },
  ml:      { bg:'rgba(168,85,247,.12)',   bd:'rgba(168,85,247,.3)',   fg:'#c084fc' },
  analytics:{ bg:'rgba(244,114,182,.12)',bd:'rgba(244,114,182,.3)',   fg:'#f472b6' },
  viz:     { bg:'rgba(34,197,94,.12)',    bd:'rgba(34,197,94,.3)',    fg:'#4ade80' },
  route:   { bg:'rgba(45,212,191,.12)',   bd:'rgba(45,212,191,.3)',   fg:'#2dd4bf' },
  default: { bg:'rgba(251,191,36,.12)',   bd:'rgba(251,191,36,.3)',   fg:'#fbbf24' },
}

function ReasoningStep({ text, done, active }) {
  const cat  = Object.keys(STEP_COLORS).find(k => text.toLowerCase().includes(k)) || 'default'
  const col  = STEP_COLORS[cat]
  return (
    <div style={{ display:'flex', gap:10, padding:'6px 0', opacity: done||active ? 1 : 0.3, transition:'opacity .3s' }}>
      <div style={{
        width:22, height:22, borderRadius:6, flexShrink:0, marginTop:1,
        display:'flex', alignItems:'center', justifyContent:'center',
        background: done||active ? col.bg : 'rgba(255,255,255,.04)',
        border:`1px solid ${done||active ? col.bd : 'rgba(255,255,255,.08)'}`,
        color: done||active ? col.fg : T.muted,
      }}>
        {done ? <Ic.Check /> : active ? <Ic.Spin color={col.fg} /> : <span style={{fontSize:9}}>●</span>}
      </div>
      <p style={{ margin:0, fontSize:11.5, color: done||active ? T.text : T.muted,
        fontFamily: text.startsWith('Tool') ? T.mono : T.sans, lineHeight:1.5 }}>{text}</p>
    </div>
  )
}

// ─── StatCard ────────────────────────────────────────────────────────────────
function StatCard({ label, value, color = T.green, sub }) {
  return (
    <div style={{ ...card, flex:1, minWidth:120 }}>
      <p style={{ color:T.muted, fontSize:10, fontWeight:600, textTransform:'uppercase', letterSpacing:.8, margin:'0 0 6px' }}>{label}</p>
      <p style={{ color, fontSize:24, fontWeight:700, margin:'0 0 2px', fontVariantNumeric:'tabular-nums' }}>{value}</p>
      {sub && <p style={{ color:T.muted, fontSize:10.5, margin:0 }}>{sub}</p>}
    </div>
  )
}

// ─── Pill badge ─────────────────────────────────────────────────────────────
function Badge({ children, color = T.green }) {
  return (
    <span style={{ padding:'2px 8px', borderRadius:5, fontSize:10.5, fontWeight:600,
      background:`${color}22`, color, border:`1px solid ${color}44` }}>
      {children}
    </span>
  )
}

// ─── Tab button ─────────────────────────────────────────────────────────────
function TabBtn({ icon, label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      display:'flex', alignItems:'center', gap:6,
      padding:'6px 14px', borderRadius:8, border:'none', cursor:'pointer',
      background: active ? T.greenDim : 'transparent',
      color: active ? T.green : T.muted,
      fontSize:12.5, fontWeight: active ? 600 : 400,
      transition:'all .2s',
    }}>
      {icon}{label}
    </button>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 1 — QUERY (Chat interface connected to real API)
// ════════════════════════════════════════════════════════════════════════════
function QueryTab({ onChartAdded }) {
  const [messages, setMessages]     = useState([])
  const [input, setInput]           = useState('')
  const [loading, setLoading]       = useState(false)
  const [examples, setExamples]     = useState([])
  const endRef = useRef(null)

  useEffect(() => { getExamples().then(d => setExamples(d.examples || [])).catch(() => {}) }, [])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior:'smooth' }) }, [messages, loading])

  const send = useCallback(async (text) => {
    const q = (text || input).trim()
    if (!q || loading) return
    setInput('')
    setLoading(true)
    setMessages(prev => [...prev, { role:'user', text: q }])

    try {
      const res = await queryAgent(q)
      setMessages(prev => [...prev, { role:'agent', ...res }])
      if (res.charts?.length) onChartAdded(res.charts)
    } catch {
      setMessages(prev => [...prev, { role:'agent', answer:'API unavailable — is the backend running? (`uvicorn src.api.main:app --reload`)', reasoning_trace:[], charts:[] }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, onChartAdded])

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100%', overflow:'hidden' }}>
      {/* Chat area */}
      <div style={{ flex:1, overflowY:'auto', padding:'20px 24px', display:'flex', flexDirection:'column', gap:16 }}>
        {messages.length === 0 && !loading && (
          <div style={{ margin:'auto', textAlign:'center', animation:'fadeUp .5s ease' }}>
            <div style={{ width:60, height:60, borderRadius:16, background:'linear-gradient(135deg,rgba(16,185,129,.15),rgba(59,130,246,.15))', border:`1px solid ${T.greenBd}`, display:'flex', alignItems:'center', justifyContent:'center', margin:'0 auto 16px' }}>
              <span style={{ color:T.green }}><Ic.Leaf /></span>
            </div>
            <h2 style={{ fontSize:18, fontWeight:600, color:T.text, marginBottom:6 }}>Ask your supply chain anything</h2>
            <p style={{ fontSize:13, color:T.muted, marginBottom:20 }}>AgriFlow reasons across USDA data, crop statistics, community indicators, and ML models</p>
            <div style={{ display:'flex', flexWrap:'wrap', gap:8, justifyContent:'center', maxWidth:660 }}>
              {examples.slice(0,6).map((ex, i) => (
                <button key={i} onClick={() => send(ex.query)} style={{
                  padding:'9px 14px', borderRadius:10, background:T.surface, border:`1px solid ${T.border}`,
                  color:T.sub, fontSize:12, cursor:'pointer', textAlign:'left', maxWidth:300,
                  transition:'all .2s',
                }}
                onMouseEnter={e => { e.currentTarget.style.background=T.greenDim; e.currentTarget.style.borderColor=T.greenBd; e.currentTarget.style.color=T.text }}
                onMouseLeave={e => { e.currentTarget.style.background=T.surface; e.currentTarget.style.borderColor=T.border; e.currentTarget.style.color=T.sub }}
                >
                  <span style={{ display:'block', fontWeight:600, marginBottom:2, color:T.text, fontSize:11.5 }}>{ex.title}</span>
                  {ex.query}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} style={{ animation:'fadeUp .3s ease' }}>
            {msg.role === 'user' ? (
              <div style={{ display:'flex', justifyContent:'flex-end' }}>
                <div style={{ maxWidth:'70%', padding:'10px 16px', borderRadius:'14px 14px 4px 14px', background:'linear-gradient(135deg,#059669,#0d9488)', color:'#f0fdf4', fontSize:13, lineHeight:1.55 }}>
                  {msg.text}
                </div>
              </div>
            ) : (
              <div style={{ display:'flex', gap:10 }}>
                <div style={{ width:28, height:28, borderRadius:8, flexShrink:0, marginTop:2, background:'linear-gradient(135deg,#059669,#10b981)', display:'flex', alignItems:'center', justifyContent:'center', color:'#f0fdf4' }}>
                  <Ic.Leaf />
                </div>
                <div style={{ flex:1 }}>
                  {/* Reasoning trace */}
                  {msg.reasoning_trace?.length > 0 && (
                    <div style={{ ...card, marginBottom:8, borderRadius:'4px 12px 12px 12px' }}>
                      <p style={{ color:T.muted, fontSize:10, fontWeight:600, textTransform:'uppercase', letterSpacing:.8, marginBottom:8 }}>Reasoning</p>
                      {msg.reasoning_trace.map((step, j) => (
                        <ReasoningStep key={j} text={step} done={true} active={false} />
                      ))}
                    </div>
                  )}
                  {/* Answer (markdown formatted) */}
                  <div style={{ ...card, borderRadius:'4px 12px 12px 12px', fontSize:13, lineHeight:1.65, color:T.text }}>
                    <Md text={msg.answer} />
                  </div>
                  {/* Inline charts */}
                  {msg.charts?.map((ch, j) => (
                    <div key={j} style={{ ...card, marginTop:8, borderRadius:12 }}>
                      <p style={{ color:T.muted, fontSize:10.5, fontWeight:600, marginBottom:8 }}>{ch.plotly_spec?.layout?.title || `Chart ${j+1}`}</p>
                      <PlotlyChart spec={ch.plotly_spec} />
                    </div>
                  ))}
                  {/* Analytics metrics */}
                  {msg.analytics_report?.evaluation && (
                    <div style={{ ...card, marginTop:8, borderRadius:12, display:'flex', gap:12, flexWrap:'wrap' }}>
                      {Object.entries(msg.analytics_report.evaluation).map(([k,v]) => (
                        <div key={k} style={{ textAlign:'center' }}>
                          <p style={{ color:T.green, fontSize:18, fontWeight:700, margin:0 }}>{typeof v === 'number' ? v.toFixed(3) : v}</p>
                          <p style={{ color:T.muted, fontSize:10, fontWeight:600, textTransform:'uppercase', margin:0 }}>{k}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Loading state */}
        {loading && (
          <div style={{ display:'flex', gap:10, animation:'fadeUp .3s ease' }}>
            <div style={{ width:28, height:28, borderRadius:8, background:'linear-gradient(135deg,#059669,#10b981)', display:'flex', alignItems:'center', justifyContent:'center', color:'#f0fdf4' }}>
              <Ic.Leaf />
            </div>
            <div style={{ ...card, borderRadius:'4px 12px 12px 12px', display:'flex', alignItems:'center', gap:10 }}>
              <Ic.Spin />
              <span style={{ color:T.sub, fontSize:12 }}>Reasoning across data sources...</span>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Quick queries */}
      <div style={{ padding:'8px 24px 0', display:'flex', gap:6, flexWrap:'wrap', flexShrink:0 }}>
        {['Food Insecurity Hotspots','Corn Dependency Analysis','Drought Scenario Planning','Route Optimization'].map((label, i) => (
          <button key={i} onClick={() => examples[i] && send(examples[i].query)} disabled={loading} style={{
            padding:'5px 10px', borderRadius:7, border:`1px solid ${T.border}`,
            background:T.surface, color: loading ? T.muted : T.sub, fontSize:11, cursor: loading ? 'default':'pointer',
            transition:'all .2s',
          }}
          onMouseEnter={e => { if(!loading) { e.currentTarget.style.background=T.greenDim; e.currentTarget.style.borderColor=T.greenBd; e.currentTarget.style.color=T.text }}}
          onMouseLeave={e => { e.currentTarget.style.background=T.surface; e.currentTarget.style.borderColor=T.border; e.currentTarget.style.color=loading?T.muted:T.sub }}
          >{label}</button>
        ))}
        {messages.length > 0 && (
          <button onClick={() => setMessages([])} style={{ marginLeft:'auto', padding:'5px 10px', borderRadius:7, border:`1px solid ${T.border}`, background:T.surface, color:T.muted, fontSize:11, cursor:'pointer' }}>
            Clear
          </button>
        )}
      </div>

      {/* Input */}
      <div style={{ padding:'12px 24px 16px', flexShrink:0 }}>
        <div style={{ display:'flex', gap:8, background:'rgba(255,255,255,.04)', border:`1px solid ${T.borderHi}`, borderRadius:12, padding:'4px 4px 4px 14px' }}>
          <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key==='Enter'&&!e.shiftKey&&send()} disabled={loading}
            placeholder="Ask about food supply chains, crop data, route optimization, risk predictions..."
            style={{ flex:1, border:'none', outline:'none', background:'transparent', color:T.text, fontSize:13 }} />
          <button onClick={() => send()} disabled={loading||!input.trim()} style={{
            width:36, height:36, borderRadius:9, border:'none', cursor: input.trim()&&!loading ? 'pointer':'default',
            background: input.trim()&&!loading ? 'linear-gradient(135deg,#059669,#10b981)' : 'rgba(255,255,255,.05)',
            color: input.trim()&&!loading ? '#f0fdf4' : T.muted,
            display:'flex', alignItems:'center', justifyContent:'center', transition:'all .2s',
          }}>
            <Ic.Send />
          </button>
        </div>
        <p style={{ color:T.muted, fontSize:10, marginTop:6 }}>
          Powered by USDA Food Environment Atlas · Food Access Research Atlas · NASS Quick Stats · FEMA · Census ACS · Open-Meteo
        </p>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 2 — DASHBOARD
// ════════════════════════════════════════════════════════════════════════════
function DashboardTab({ sessionCharts }) {
  const [analytics, setAnalytics] = useState([])
  const [evalData,  setEvalData]  = useState(null)

  useEffect(() => {
    getAnalytics().then(d => setAnalytics(d.reports || [])).catch(() => {})
    getEvalSummary().then(setEvalData).catch(() => {})
  }, [sessionCharts])

  const latestReport = analytics[analytics.length - 1]

  return (
    <div style={{ overflowY:'auto', padding:24, display:'flex', flexDirection:'column', gap:20 }}>
      {/* Stat row */}
      <div style={{ display:'flex', gap:12, flexWrap:'wrap' }}>
        <StatCard label="Charts Generated"    value={sessionCharts.length}        color={T.green}  sub="This session" />
        <StatCard label="Analytics Reports"   value={analytics.length}             color={T.blue}   sub="ML pipelines run" />
        <StatCard label="Avg Quality Score"   value={evalData?.avg_score ?? '—'}   color={T.purple} sub="Response quality" />
        <StatCard label="Avg Tool Calls"      value={evalData?.avg_tool_calls ?? '—'} color={T.warn} sub="Per query" />
      </div>

      {/* Session charts */}
      {sessionCharts.length > 0 && (
        <div>
          <p style={{ color:T.muted, fontSize:11, fontWeight:600, textTransform:'uppercase', letterSpacing:.8, marginBottom:12 }}>Session Charts</p>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(440px,1fr))', gap:16 }}>
            {sessionCharts.map((ch, i) => (
              <div key={i} style={card}>
                <p style={{ color:T.sub, fontSize:12, fontWeight:600, marginBottom:8 }}>
                  {ch.plotly_spec?.layout?.title || `Chart ${i+1}`}
                </p>
                <PlotlyChart spec={ch.plotly_spec} height={240} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Latest ML analytics report */}
      {latestReport && (
        <div style={card}>
          <p style={{ color:T.muted, fontSize:11, fontWeight:600, textTransform:'uppercase', letterSpacing:.8, marginBottom:12 }}>Latest ML Analytics Report</p>
          <div style={{ display:'flex', gap:12, flexWrap:'wrap', marginBottom:16 }}>
            {latestReport.evaluation && Object.entries(latestReport.evaluation).map(([k,v]) => (
              <div key={k} style={{ background:T.greenDim, border:`1px solid ${T.greenBd}`, borderRadius:8, padding:'8px 14px', textAlign:'center' }}>
                <p style={{ color:T.green, fontSize:22, fontWeight:700, margin:0, fontVariantNumeric:'tabular-nums' }}>{typeof v==='number'?v.toFixed(3):v}</p>
                <p style={{ color:T.muted, fontSize:10, fontWeight:600, textTransform:'uppercase', margin:0 }}>{k}</p>
              </div>
            ))}
          </div>
          {latestReport.feature_importance?.slice(0,5).map((f, i) => (
            <div key={i} style={{ display:'flex', alignItems:'center', gap:10, marginBottom:6 }}>
              <div style={{ width:70, fontSize:10.5, color:T.muted, textAlign:'right', flexShrink:0 }}>{f.feature?.split('_').slice(-1)[0]}</div>
              <div style={{ flex:1, height:6, background:'rgba(255,255,255,.06)', borderRadius:3, overflow:'hidden' }}>
                <div style={{ width:`${(f.importance||0)*100}%`, height:'100%', background:`linear-gradient(90deg,${T.green},${T.blue})`, borderRadius:3 }} />
              </div>
              <div style={{ width:36, fontSize:10.5, color:T.sub }}>{((f.importance||0)*100).toFixed(1)}%</div>
            </div>
          ))}
          {latestReport.analysis_report && (
            <p style={{ color:T.sub, fontSize:12, lineHeight:1.6, marginTop:12, whiteSpace:'pre-wrap' }}>{latestReport.analysis_report}</p>
          )}
        </div>
      )}

      {/* Quality issues */}
      {evalData?.top_issues?.length > 0 && (
        <div style={card}>
          <p style={{ color:T.muted, fontSize:11, fontWeight:600, textTransform:'uppercase', letterSpacing:.8, marginBottom:12 }}>Top Improvement Areas <Badge color={T.warn}>Auto-detected</Badge></p>
          {evalData.top_issues.map((issue, i) => (
            <div key={i} style={{ display:'flex', gap:10, alignItems:'flex-start', padding:'8px 0', borderBottom:`1px solid ${T.border}` }}>
              <span style={{ color:T.warn, fontSize:10, fontWeight:700, minWidth:16 }}>{i+1}.</span>
              <p style={{ margin:0, fontSize:12, color:T.sub, flex:1 }}>{issue.issue}</p>
              <Badge color={T.muted}>{issue.count}×</Badge>
            </div>
          ))}
        </div>
      )}

      {sessionCharts.length === 0 && analytics.length === 0 && (
        <div style={{ textAlign:'center', padding:'60px 20px', color:T.muted }}>
          <p style={{ fontSize:32, marginBottom:12 }}>📊</p>
          <p style={{ fontSize:15, fontWeight:600, color:T.sub, marginBottom:6 }}>No data yet</p>
          <p style={{ fontSize:13 }}>Run queries in the Query tab to generate charts and analytics reports here.</p>
        </div>
      )}
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 3 — DATA SOURCES
// ════════════════════════════════════════════════════════════════════════════
const SOURCE_META = [
  { name:'USDA Food Environment Atlas', cat:'data',  type:'SQLite',    key:null,          desc:'County-level food insecurity, SNAP, poverty (307 cols, 3,156 counties)' },
  { name:'USDA Food Access Research Atlas', cat:'data', type:'SQLite', key:null,          desc:'Census-tract food desert classification, LILA flags (121 cols, 72K tracts)' },
  { name:'USDA NASS Quick Stats',       cat:'data',  type:'REST API',  key:'nass',        desc:'County crop yields, production, acreage — live API with 30+ years of data' },
  { name:'FEMA Disaster Declarations',  cat:'data',  type:'REST API',  key:null,          desc:'Historical floods, storms, droughts by county (free, no key required)' },
  { name:'US Census ACS',               cat:'data',  type:'REST API',  key:null,          desc:'Demographics, income, poverty, vehicle access, unemployment (free)' },
  { name:'Open-Meteo Weather',          cat:'data',  type:'REST API',  key:null,          desc:'7-day forecasts for Missouri counties — temperature, precipitation (free)' },
  { name:'DuckDuckGo Web Search',       cat:'data',  type:'REST API',  key:null,          desc:'Emerging agricultural threats, pest/disease alerts (free)' },
  { name:'Archia ODBC',                 cat:'sql',   type:'MCP/HTTP',  key:'archia',      desc:'Server-side ODBC SQL access via Archia platform tool' },
  { name:'AgriFlow SQLite (agriflow.db)', cat:'sql', type:'Local',     key:null,          desc:'Unified database with loaded USDA datasets and any ingested tables' },
]

const CAT_COLOR = { data:T.blue, sql:T.purple, ml:T.warn, route:'#2dd4bf' }

function DataSourcesTab() {
  const [health, setHealth] = useState(null)

  useEffect(() => { getHealth().then(setHealth).catch(() => {}) }, [])

  const isKeyOk = (keyName) => {
    if (!keyName || !health) return true
    if (keyName === 'nass')   return health.nass_configured
    if (keyName === 'archia') return health.archia_configured
    return true
  }

  return (
    <div style={{ overflowY:'auto', padding:24, display:'flex', flexDirection:'column', gap:20 }}>
      {/* Status bar */}
      <div style={{ display:'flex', gap:12, flexWrap:'wrap' }}>
        <StatCard label="API Status"    value={health ? 'Online' : '—'} color={health ? T.green : T.muted}  sub={`v${health?.model?.split('-').pop() || '—'}`} />
        <StatCard label="Total Tools"   value={health?.total_tools ?? '—'} color={T.blue}   sub="Across all categories" />
        <StatCard label="NASS API"      value={health?.nass_configured ? 'Active' : 'No Key'} color={health?.nass_configured ? T.green : T.warn} sub="Required for crop data" />
        <StatCard label="Archia Cloud"  value={health?.archia_configured ? 'Active' : 'No Token'} color={health?.archia_configured ? T.green : T.muted} sub="Agent orchestration" />
      </div>

      {/* Data source table */}
      <div style={card}>
        <p style={{ color:T.muted, fontSize:11, fontWeight:600, textTransform:'uppercase', letterSpacing:.8, marginBottom:14 }}>Connected Data Sources</p>
        <table style={{ width:'100%', borderCollapse:'collapse', fontSize:12 }}>
          <thead>
            <tr>
              {['Source', 'Category', 'Access', 'Status', 'Description'].map(h => (
                <th key={h} style={{ padding:'8px 12px', textAlign:'left', color:T.muted, fontWeight:600, fontSize:10.5, textTransform:'uppercase', letterSpacing:.5, borderBottom:`1px solid ${T.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SOURCE_META.map((src, i) => {
              const ok = isKeyOk(src.key)
              return (
                <tr key={i} style={{ background: i%2===0 ? 'rgba(255,255,255,.015)' : 'transparent' }}>
                  <td style={{ padding:'9px 12px', color:T.text, fontWeight:500, borderBottom:`1px solid ${T.border}`, whiteSpace:'nowrap' }}>{src.name}</td>
                  <td style={{ padding:'9px 12px', borderBottom:`1px solid ${T.border}` }}>
                    <Badge color={CAT_COLOR[src.cat] || T.blue}>{src.cat}</Badge>
                  </td>
                  <td style={{ padding:'9px 12px', color:src.type==='REST API'?T.warn:T.sub, fontSize:11.5, borderBottom:`1px solid ${T.border}`, whiteSpace:'nowrap' }}>{src.type}</td>
                  <td style={{ padding:'9px 12px', borderBottom:`1px solid ${T.border}` }}>
                    <div style={{ display:'flex', alignItems:'center', gap:5 }}>
                      <div style={{ width:6, height:6, borderRadius:'50%', background: ok ? T.green : T.warn, boxShadow: ok ? `0 0 6px ${T.green}66` : 'none' }} />
                      <span style={{ fontSize:11, color: ok ? '#6ee7b7' : T.warn }}>{ok ? 'Ready' : 'Config required'}</span>
                    </div>
                  </td>
                  <td style={{ padding:'9px 12px', color:T.muted, fontSize:11.5, borderBottom:`1px solid ${T.border}`, lineHeight:1.4 }}>{src.desc}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Tool breakdown */}
      {health?.tools_available && (
        <div style={card}>
          <p style={{ color:T.muted, fontSize:11, fontWeight:600, textTransform:'uppercase', letterSpacing:.8, marginBottom:14 }}>Tool Inventory ({health.total_tools} total)</p>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(200px,1fr))', gap:10 }}>
            {Object.entries(health.tools_available).map(([cat, tools]) => (
              <div key={cat} style={{ background:T.surface, border:`1px solid ${T.border}`, borderRadius:8, padding:12 }}>
                <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:8 }}>
                  <Badge color={CAT_COLOR[cat] || T.blue}>{cat}</Badge>
                  <span style={{ color:T.muted, fontSize:10 }}>{tools.length} tools</span>
                </div>
                {tools.map(t => (
                  <p key={t} style={{ margin:'2px 0', fontSize:10.5, color:T.muted, fontFamily:T.mono }}>{t}</p>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 4 — MAP & ALERTS
// ════════════════════════════════════════════════════════════════════════════
const ALERT_QUERIES = [
  { id:'disaster', label:'FEMA Disaster Alerts', icon:'⚠️', color:T.danger,
    query:'What FEMA disaster declarations have affected Missouri counties since 2022? List the most recent events with affected counties and disaster types.' },
  { id:'disease',  label:'Crop Disease & Pest Watch', icon:'🦠', color:T.warn,
    query:'Search the web for emerging agricultural threats, crop diseases, and pest outbreaks affecting Missouri and surrounding states in 2026.' },
]

const MO_LOCATIONS = [
  'St. Louis Food Bank','Springfield Hub','Cape Girardeau Hub','Columbia Hub','Joplin Hub',
  'Sikeston Hub','Wayne County','Pemiscot County','Dunklin County','New Madrid County',
  'Oregon County','Ozark County','Ripley County','Shannon County','Howell County',
]

function AlertsTab({ onChartAdded }) {
  const [alertResults, setAlertResults] = useState({})
  const [alertLoading, setAlertLoading] = useState({})
  const [origin, setOrigin]             = useState('Cape Girardeau Hub')
  const [dests, setDests]               = useState('Wayne County, Pemiscot County, Dunklin County')
  const [routeResult, setRouteResult]   = useState(null)
  const [routeLoading, setRouteLoading] = useState(false)

  const runAlert = async (alert) => {
    setAlertLoading(p => ({ ...p, [alert.id]: true }))
    try {
      const res = await queryAgent(alert.query)
      setAlertResults(p => ({ ...p, [alert.id]: res }))
      if (res.charts?.length) onChartAdded(res.charts)
    } catch {
      setAlertResults(p => ({ ...p, [alert.id]: { answer:'API unavailable.', charts:[] } }))
    } finally {
      setAlertLoading(p => ({ ...p, [alert.id]: false }))
    }
  }

  const planRoute = async () => {
    if (!origin || !dests.trim() || routeLoading) return
    setRouteLoading(true)
    setRouteResult(null)
    try {
      const q = `Plan an optimal delivery route from ${origin} to ${dests}. Show the route on a map and provide a delivery schedule starting at 08:00.`
      const res = await queryAgent(q)
      setRouteResult(res)
      if (res.charts?.length) onChartAdded(res.charts)
    } catch {
      setRouteResult({ answer:'API unavailable.', charts:[] })
    } finally {
      setRouteLoading(false)
    }
  }

  return (
    <div style={{ overflowY:'auto', padding:24, display:'flex', flexDirection:'column', gap:20 }}>
      {/* Route planner */}
      <div style={card}>
        <p style={{ color:T.muted, fontSize:11, fontWeight:600, textTransform:'uppercase', letterSpacing:.8, marginBottom:14 }}>
          <span style={{ color:'#2dd4bf' }}>◈</span> Delivery Route Optimizer
        </p>
        <div style={{ display:'flex', gap:12, flexWrap:'wrap', marginBottom:12 }}>
          <div style={{ flex:1, minWidth:200 }}>
            <label style={{ display:'block', color:T.muted, fontSize:10.5, fontWeight:600, marginBottom:6, textTransform:'uppercase', letterSpacing:.5 }}>Origin</label>
            <select value={origin} onChange={e => setOrigin(e.target.value)} style={{
              width:'100%', padding:'8px 12px', borderRadius:8, border:`1px solid ${T.border}`,
              background:'rgba(255,255,255,.05)', color:T.text, fontSize:12.5, outline:'none',
            }}>
              {MO_LOCATIONS.slice(0,6).map(loc => <option key={loc} value={loc} style={{ background:'#1a1f2e' }}>{loc}</option>)}
            </select>
          </div>
          <div style={{ flex:2, minWidth:280 }}>
            <label style={{ display:'block', color:T.muted, fontSize:10.5, fontWeight:600, marginBottom:6, textTransform:'uppercase', letterSpacing:.5 }}>Destinations (comma-separated)</label>
            <input value={dests} onChange={e => setDests(e.target.value)}
              placeholder="Wayne County, Pemiscot County, Dunklin County"
              style={{ width:'100%', padding:'8px 12px', borderRadius:8, border:`1px solid ${T.border}`, background:'rgba(255,255,255,.05)', color:T.text, fontSize:12.5, outline:'none' }} />
          </div>
        </div>
        <button onClick={planRoute} disabled={routeLoading} style={{
          padding:'9px 20px', borderRadius:9, border:'none', cursor: routeLoading ? 'default':'pointer',
          background: routeLoading ? 'rgba(255,255,255,.05)' : 'linear-gradient(135deg,#0d9488,#0891b2)',
          color: routeLoading ? T.muted : '#f0fdf4', fontSize:13, fontWeight:600, display:'flex', alignItems:'center', gap:8, transition:'all .2s',
        }}>
          {routeLoading ? <><Ic.Spin color={T.sub} /> Optimizing...</> : '⟳ Optimize Route'}
        </button>

        {routeResult && (
          <div style={{ marginTop:16 }}>
            <p style={{ color:T.text, fontSize:12.5, lineHeight:1.6, whiteSpace:'pre-wrap', marginBottom:12 }}>{routeResult.answer}</p>
            {routeResult.charts?.map((ch, i) => (
              <div key={i} style={{ marginTop:8 }}>
                <PlotlyChart spec={ch.plotly_spec} height={320} />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Alert panels */}
      {ALERT_QUERIES.map(alert => (
        <div key={alert.id} style={{ ...card, borderColor: alertResults[alert.id] ? `${alert.color}33` : T.border }}>
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:12 }}>
            <p style={{ color:T.muted, fontSize:11, fontWeight:600, textTransform:'uppercase', letterSpacing:.8, margin:0 }}>
              {alert.icon} {alert.label}
            </p>
            <button onClick={() => runAlert(alert)} disabled={alertLoading[alert.id]} style={{
              padding:'6px 14px', borderRadius:8, border:`1px solid ${alert.color}44`,
              background:`${alert.color}11`, color:alert.color, fontSize:11.5, fontWeight:600,
              cursor: alertLoading[alert.id] ? 'default':'pointer', display:'flex', alignItems:'center', gap:6,
            }}>
              {alertLoading[alert.id] ? <><Ic.Spin color={alert.color} /> Fetching...</> : '↻ Refresh'}
            </button>
          </div>

          {alertResults[alert.id] ? (
            <div>
              <p style={{ color:T.sub, fontSize:12.5, lineHeight:1.65, whiteSpace:'pre-wrap' }}>{alertResults[alert.id].answer}</p>
              {alertResults[alert.id].charts?.map((ch, i) => (
                <div key={i} style={{ marginTop:10 }}>
                  <PlotlyChart spec={ch.plotly_spec} height={260} />
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color:T.muted, fontSize:12, fontStyle:'italic' }}>Click Refresh to fetch live data via the AgriFlow agent.</p>
          )}
        </div>
      ))}
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// ROOT APP
// ════════════════════════════════════════════════════════════════════════════
const TABS = [
  { id:'query',   label:'Query',        Icon: Ic.Chat },
  { id:'dash',    label:'Dashboard',    Icon: Ic.Grid },
  { id:'sources', label:'Data Sources', Icon: Ic.Db   },
  { id:'alerts',  label:'Map & Alerts', Icon: Ic.Map  },
]

export default function App() {
  const [tab, setTab]             = useState('query')
  const [charts, setCharts]       = useState([])

  const addCharts = useCallback((newCharts) => {
    setCharts(prev => {
      const ids = new Set(prev.map(c => c.chart_id))
      return [...prev, ...newCharts.filter(c => !ids.has(c.chart_id))]
    })
  }, [])

  return (
    <div style={{ width:'100%', height:'100vh', display:'flex', flexDirection:'column', background:T.bg, fontFamily:T.sans, color:T.text, overflow:'hidden' }}>
      <style>{GLOBAL_CSS}</style>

      {/* Header */}
      <header style={{ display:'flex', alignItems:'center', gap:16, padding:'10px 20px', borderBottom:`1px solid ${T.border}`, background:'rgba(255,255,255,.015)', flexShrink:0 }}>
        {/* Logo */}
        <div style={{ display:'flex', alignItems:'center', gap:9 }}>
          <div style={{ width:32, height:32, borderRadius:8, background:'linear-gradient(135deg,#059669,#10b981)', display:'flex', alignItems:'center', justifyContent:'center', boxShadow:'0 0 18px rgba(16,185,129,.3)', color:'#f0fdf4' }}>
            <Ic.Leaf />
          </div>
          <div>
            <div style={{ display:'flex', alignItems:'baseline', gap:5 }}>
              <span style={{ fontSize:16, fontWeight:700, color:'#f0fdf4', letterSpacing:-.3 }}>AgriFlow</span>
              <span style={{ fontSize:10, fontWeight:600, color:T.green, letterSpacing:1, textTransform:'uppercase' }}>Agent</span>
            </div>
            <span style={{ fontSize:10.5, color:T.muted }}>Food Supply Chain Intelligence</span>
          </div>
        </div>

        {/* Tabs */}
        <nav style={{ display:'flex', gap:2, background:'rgba(255,255,255,.04)', padding:4, borderRadius:10, marginLeft:8 }}>
          {TABS.map(t => (
            <TabBtn key={t.id} icon={<t.Icon />} label={t.label} active={tab===t.id} onClick={() => setTab(t.id)} />
          ))}
        </nav>

        {/* Status */}
        <div style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:10 }}>
          {charts.length > 0 && (
            <span style={{ fontSize:11, color:T.muted }}>{charts.length} chart{charts.length!==1?'s':''} generated</span>
          )}
          <div style={{ display:'flex', alignItems:'center', gap:5 }}>
            <div style={{ width:6, height:6, borderRadius:'50%', background:T.green, boxShadow:`0 0 8px ${T.green}88` }} />
            <span style={{ fontSize:11, color:T.muted }}>AgriFlow v2.1</span>
          </div>
          <div style={{ padding:'3px 10px', borderRadius:6, background:T.greenDim, border:`1px solid ${T.greenBd}` }}>
            <span style={{ fontSize:10.5, color:'#6ee7b7', fontWeight:500 }}>MUIDSI 2026</span>
          </div>
        </div>
      </header>

      {/* Tab content — all tabs stay mounted so state persists across switches */}
      <main style={{ flex:1, overflow:'hidden', position:'relative' }}>
        <div style={{ display: tab==='query'  ?'flex':'none', flexDirection:'column', height:'100%' }}><QueryTab   onChartAdded={addCharts} /></div>
        <div style={{ display: tab==='dash'   ?'flex':'none', flexDirection:'column', height:'100%' }}><DashboardTab sessionCharts={charts} /></div>
        <div style={{ display: tab==='sources'?'flex':'none', flexDirection:'column', height:'100%' }}><DataSourcesTab /></div>
        <div style={{ display: tab==='alerts' ?'flex':'none', flexDirection:'column', height:'100%' }}><AlertsTab onChartAdded={addCharts} /></div>
      </main>
    </div>
  )
}
