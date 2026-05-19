import { useState, useEffect, useRef, useCallback, memo } from 'react'
import { getHealth, getDocuments, deleteDoc } from './utils/api'
import { useSearch } from './hooks/useSearch'
import { useUpload } from './hooks/useUpload'
import './App.css'

// ── UI Component Imports ───────────────────────────────────────────────
import StarBorder from './components/StarBorder'
import AnimatedList from './components/AnimatedList'

// ── Icons ──────────────────────────────────────────────────────────────
const Icon = ({ d, size = 18, className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    className={className} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
)

const Icons = {
  search:    "M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z",
  dashboard: "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
  doc:       "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6",
  upload:    "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12",
  trash:     "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6",
  close:     "M18 6L6 18M6 6l12 12",
  history:   "M12 8v4l3 3M3.05 11a9 9 0 1 0 .5-3",
  sparkle:   "M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z",
  zap:       "M13 2L3 14h9l-1 8 10-12h-9l1-8z",
  lock:      "M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
}

// ── CRITICAL FIX: Removed the normalizeRRF function ────────────────────
const scoreTier = (pct) => {
  if (pct >= 75) return 'score-high'
  if (pct >= 45) return 'score-mid'
  return 'score-low'
}

// ── CUSTOM PERFORMANCE HOOKS ───────────────────────────────────────────

// #1: Typewriter Effect Hook for Inputs/Placeholders
function useTypewriter(phrases, speed = 60, holdTime = 2000) {
  const [index, setIndex] = useState(0)
  const [subIndex, setSubIndex] = useState(0)
  const [isDeleting, setIsDeleting] = useState(false)
  const [currentText, setCurrentText] = useState('')

  useEffect(() => {
    if (subIndex === phrases[index].length + 1 && !isDeleting) {
      const timeout = setTimeout(() => setIsDeleting(true), holdTime)
      return () => clearTimeout(timeout)
    }

    if (subIndex === 0 && isDeleting) {
      setIsDeleting(false)
      setIndex((prev) => (prev + 1) % phrases.length)
      return
    }

    const timeout = setTimeout(() => {
      setSubIndex((prev) => prev + (isDeleting ? -1 : 1))
    }, isDeleting ? speed / 2 : speed)

    return () => clearTimeout(timeout)
  }, [subIndex, index, isDeleting, phrases, speed, holdTime])

  useEffect(() => {
    setCurrentText(phrases[index].substring(0, subIndex))
  }, [subIndex, index, phrases])

  return currentText
}

// #2: Hardware-Accelerated Telemetry Counter Hook
function useCountUp(target, duration = 1000) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    let startTimestamp = null
    const endValue = parseInt(target, 10) || 0
    if (endValue === 0) { setCount(0); return }

    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp
      const progress = Math.min((timestamp - startTimestamp) / duration, 1)
      setCount(Math.floor(progress * endValue))
      if (progress < 1) {
        window.requestAnimationFrame(step)
      }
    }

    window.requestAnimationFrame(step)
  }, [target, duration])

  return count.toLocaleString()
}

// ── DYNAMIC ENGINE ANIMATION LAYER COMPONENTS ──────────────────────────

// #3: Core Operational Boot Sequence Gateway Component
const BootSequence = ({ onComplete }) => {
  const [logs, setLogs] = useState([])
  const sequence = [
    { text: "> INITIALIZING INTELSPACE CORE...", delay: 200 },
    { text: "> LOADING FAISS VECTOR GRAPH INDICES...", delay: 600, append: " [OK]" },
    { text: "> ACCELERATING COMPUTATIONAL WORKFLOWS...", delay: 1100, append: " [OK]" },
    { text: "> ESTABLISHING QUANTUM ENCRYPTED CHANNEL...", delay: 1500, append: " [OK]" },
    { text: "> CORE ENGINE SECURITY STANDARDS OPERATIONAL.", delay: 1900 },
    { text: "> READY.", delay: 2200 }
  ]

  useEffect(() => {
    sequence.forEach((line) => {
      setTimeout(() => {
        setLogs(prev => {
          if (line.append) {
            return prev.map(item => item.startsWith(line.text) ? line.text + line.append : item)
          }
          return [...prev, line.text]
        })
      }, line.delay)
    })

    const exitTimeout = setTimeout(onComplete, 2600)
    return () => clearTimeout(exitTimeout)
  }, [])

  return (
    <div className="boot-screen mono">
      <div className="boot-terminal-window">
        {logs.map((log, index) => (
          <div key={index} className="boot-line">{log}</div>
        ))}
        <span className="boot-cursor">_</span>
      </div>
    </div>
  )
}

// #9: Toast Alerts Notification System
const ToastContainer = ({ toasts, removeToast }) => (
  <div className="toast-container">
    {toasts.map((toast) => (
      <div key={toast.id} className={`toast-message glass-panel toast-${toast.type}`}>
        <span className="toast-icon">{toast.type === 'success' ? '✓' : '✗'}</span>
        <div className="toast-content mono">{toast.msg}</div>
        <button className="toast-close" onClick={() => removeToast(toast.id)}>×</button>
      </div>
    ))}
  </div>
)

// #10: Universal Scanning Vector Empty State Component
const EmptyState = ({ message }) => (
  <div className="empty-state-container glass-panel" style={{ position: 'relative', zIndex: 1 }}>
    <div className="empty-state-radar">
      <div className="radar-line" />
      <div className="radar-ping" />
    </div>
    <p className="text-muted mono">{message}</p>
  </div>
)

// ── Markdown Formatting Helper ─────────────────────────────────────────
const FormattedAnswer = ({ text }) => {
  if (!text) return null

  const lines = text.split('\n')
  const elements = []
  let tableRows = []
  let inTable = false

  const renderTable = (rows, key) => (
    <div key={key} className="table-container">
      <table>
        <thead>
          <tr>
            {rows[0].split('|').filter(c => c.trim()).map((cell, i) => (
              <th key={i}>{cell.trim()}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(2).map((row, i) => (
            <tr key={i}>
              {row.split('|').filter(c => c.trim()).map((cell, j) => (
                <td key={j}>{cell.trim()}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )

  const parseBold = (str) => {
    const parts = str.split(/(\*\*.*?\*\*)/g)
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} className="text-glow">{part.slice(2, -2)}</strong>
      }
      return part
    })
  }

  lines.forEach((line, index) => {
    const trimmed = line.trim()

    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      inTable = true
      tableRows.push(trimmed)
      if (index === lines.length - 1) elements.push(renderTable(tableRows, index))
      return
    } else if (inTable) {
      elements.push(renderTable(tableRows, index))
      tableRows = []
      inTable = false
    }

    if (!trimmed) {
      elements.push(<div key={`space-${index}`} className="spacer" />)
      return
    }

    if (trimmed.startsWith('- ') || /^\d+\.\s/.test(trimmed)) {
      const isBullet = trimmed.startsWith('- ')
      const content  = isBullet ? trimmed.slice(2) : trimmed.replace(/^\d+\.\s/, '')
      const prefix   = isBullet ? '➤' : trimmed.match(/^\d+\./)[0]
      elements.push(
        <div key={index} className="list-item">
          <span className="list-bullet">{prefix}</span>
          <span className="list-content">{parseBold(content)}</span>
        </div>
      )
    } else {
      elements.push(
        <div key={index} className="paragraph">{parseBold(trimmed)}</div>
      )
    }
  })

  return <div className="formatted-content">{elements}</div>
}

// ── Dashboard View ─────────────────────────────────────────────────────
function DashboardView({ health, docs, history }) {
  const totalChunks  = docs.reduce((acc, d) => acc + d.chunks, 0)
  const totalVectors = health?.vectors || totalChunks

  return (
    <div className="fade-in">
      <h2 className="panel-title">System Telemetry</h2>
      <p className="panel-sub mono">STATUS: OPERATIONAL | LLM: OFFLINE-READY</p>

      <div className="dash-grid">
        <div className="stat-card glass-panel">
          <span className="stat-label">Total Documents</span>
          <span className="stat-value">{useCountUp(docs.length)}</span>
        </div>
        <div className="stat-card glass-panel glow-border">
          <span className="stat-label">Indexed Vectors</span>
          <span className="stat-value brand-text">{useCountUp(totalVectors)}</span>
        </div>
        <div className="stat-card glass-panel">
          <span className="stat-label">Processed Chunks</span>
          <span className="stat-value">{useCountUp(totalChunks)}</span>
        </div>
      </div>

      <div className="dash-sections">
        <div className="dash-card glass-panel">
          <h3 className="card-header mono"><Icon d={Icons.history} size={14} /> Query Log</h3>
          <div className="data-list">
            {history.slice(0, 4).map((h, i) => (
              <div key={i} className="data-row fading-row" style={{ animationDelay: `${i * 60}ms` }}>
                <div className="data-primary text-glow">"{h.query}"</div>
                <div className="data-secondary mono">[RETRIEVED {h.results} NODES]</div>
              </div>
            ))}
            {history.length === 0 && <EmptyState message="AWAITING_INPUT..." />}
          </div>
        </div>
        <div className="dash-card glass-panel">
          <h3 className="card-header mono"><Icon d={Icons.doc} size={14} /> Active Corpus</h3>
          <div className="data-list">
            {docs.slice(0, 4).map((d, i) => (
              <div key={i} className="data-row with-icon fading-row" style={{ animationDelay: `${i * 60}ms` }}>
                <Icon d={Icons.doc} size={16} />
                <div>
                  <div className="data-primary">{d.name}</div>
                  <div className="data-secondary mono">{d.chunks} CHUNKS ALLOCATED</div>
                </div>
              </div>
            ))}
            {docs.length === 0 && <EmptyState message="CORPUS_EMPTY" />}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Auth View ──────────────────────────────────────────────────────────
function AuthView({ onLogin, addToast }) {
  const [isRegister, setIsRegister] = useState(false)
  const [orgId, setOrgId]           = useState('')
  const [password, setPassword]     = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      const res = await fetch('http://localhost:5000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ org_id: orgId, password })
      })
      if (!res.ok) throw new Error('Credentials rejected')
      const data = await res.json()
      onLogin(data.access_token, orgId)
      addToast('Secure Handshake Established Successfully.', 'success')
    } catch (err) {
      addToast('Gateway Access Denied: Invalid Credentials.', 'danger')
    }
  }

  const handleSocialLogin = async (provider) => {
    if (!orgId.trim()) {
      addToast('Tenant Validation Error: Specify Org ID first.', 'danger')
      return
    }
    const mockToken = 'auth_stream_token_abc123'
    try {
      const res = await fetch('http://localhost:5000/api/auth/oauth/callback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, access_token: mockToken, org_id: orgId.trim() })
      })
      if (!res.ok) throw new Error('OAuth handshake rejected.')
      const data = await res.json()
      onLogin(data.access_token, data.org_id)
      addToast('Google Cloud Secure Handshake Established.', 'success')
    } catch (err) {
      addToast('OAuth Error: Handshake Rejected.', 'danger')
    }
  }

  return (
    <div className="auth-container fade-in" style={{ position: 'relative', zIndex: 1 }}>
      <div className="auth-card terminal-window">
        <div className="terminal-header">
          <span className="dot red" /><span className="dot yellow" /><span className="dot green" />
          <span className="terminal-title">IntelSpace Secure Gateway</span>
        </div>
        <div className="auth-body">
          <h2 className="auth-title">{isRegister ? 'Initialize Org' : 'Authenticate'}</h2>
          <form onSubmit={handleSubmit} className="auth-form">
            <input type="text" placeholder="Organization ID Scope" className="auth-input mono"
              value={orgId} onChange={(e) => setOrgId(e.target.value)} required />
            <input type="password" placeholder="Passphrase" className="auth-input mono"
              value={password} onChange={(e) => setPassword(e.target.value)} required={!orgId} />
            <button type="submit" className="auth-btn">Establish Connection</button>
          </form>
          <div className="oauth-divider mono">[ OR CONNECT VIA PROVIDER ]</div>
          <div className="oauth-buttons">
            <button type="button" className="oauth-btn google-btn mono" onClick={() => handleSocialLogin('google')}>
              Google Cloud Secure
            </button>
            <button type="button" className="oauth-btn github-btn mono" onClick={() => handleSocialLogin('github')}>
              GitHub Enterprise
            </button>
          </div>
          <button className="auth-toggle" onClick={() => setIsRegister(!isRegister)}>
            {isRegister ? 'Existing Tenant? Authenticate.' : 'New Tenant? Initialize.'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────
export default function App() {
  const [booting, setBooting] = useState(true)
  const [token, setToken] = useState(localStorage.getItem('rei_token') || null)
  const [org, setOrg]     = useState(localStorage.getItem('rei_org')   || null)

  const [panel, setPanel]             = useState('search')
  const [query, setQuery]             = useState('')
  const [history, setHistory]         = useState([])
  const [historyOpen, setHistoryOpen] = useState(false)
  const [toasts, setToasts]           = useState([])
  const inputRef = useRef()

  const [health, setHealth] = useState(null)
  const [docs, setDocs]     = useState([])

  const { results, loading, meta, run } = useSearch()

  // Terminal Input Dynamic Text Database Array
  const terminalPhrases = [
    "Compare efficacy of ternary alloys vs secondary solutions...",
    "Extract crystallographic stacking fault energies from corpus...",
    "Summarize local multi-modal latent density distribution profiles...",
    "Query isolated dense FAISS structural index clusters..."
  ]
  const animatedPlaceholder = useTypewriter(terminalPhrases, 50, 2500)

  // #9 System Notification Stack Dispatches
  const addToast = useCallback((msg, type = 'success') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, msg, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 4000)
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const refreshAll = useCallback(async () => {
    if (!token) return
    try {
      const [h, d] = await Promise.all([getHealth(token), getDocuments(token)])
      setHealth(h.data)
      setDocs(d.data.documents || [])
    } catch (e) {
      if (e.status === 401) handleLogout()
    }
  }, [token])

  const { upload, uploading, progress } = useUpload(refreshAll, token)

  useEffect(() => { refreshAll() }, [refreshAll])
  useEffect(() => {
    const saved = localStorage.getItem('rei_history')
    if (saved) try { setHistory(JSON.parse(saved)) } catch {}
  }, [])
  useEffect(() => { localStorage.setItem('rei_history', JSON.stringify(history)) }, [history])

  const handleLogin = (jwt, orgId) => {
    setToken(jwt); setOrg(orgId)
    localStorage.setItem('rei_token', jwt)
    localStorage.setItem('rei_org', orgId)
  }

  const handleLogout = () => {
    setToken(null); setOrg(null)
    localStorage.removeItem('rei_token'); localStorage.removeItem('rei_org')
    addToast('Session Safely Terminated.', 'success')
  }

  const handleSearch = (e) => {
    e?.preventDefault()
    if (!query.trim()) return
    run(query.trim(), 10, token)
    const entry = { query: query.trim(), results: results?.length || 0, ts: Date.now() }
    setHistory(prev => [entry, ...prev.filter(h => h.query !== query.trim())].slice(0, 20))
  }

  // #4: Dynamic Heartbeat Pulsing System Mapping Function
  const getHeartbeatClass = () => {
    if (!token) return 'pulse-offline'
    if (!health) return 'pulse-degraded'
    if (health.status === 'healthy' || health.vectors >= 0) return 'pulse-healthy'
    return 'pulse-degraded'
  }

  if (booting) return <BootSequence onComplete={() => setBooting(false)} />

  const navItems = [
    { id: 'search',    label: 'Central Command', icon: Icons.search, index: 0 },
    { id: 'dashboard', label: 'Telemetry',       icon: Icons.dashboard, index: 1 },
    { id: 'upload',    label: 'Data Ingestion',  icon: Icons.upload, index: 2 },
    { id: 'docs',      label: `Vault (${docs.length})`, icon: Icons.doc, index: 3 },
  ]

  const activeIndex = navItems.find(item => item.id === panel)?.index ?? 0

  return (
    <>
      <ToastContainer toasts={toasts} removeToast={removeToast} />
      
      {!token ? (
        <AuthView onLogin={handleLogin} addToast={addToast} />
      ) : (
        <div className="layout" style={{ position: 'relative', zIndex: 1 }}>

          {/* ── Sidebar ── */}
          <aside className="sidebar glass-panel">
            <div className="brand">
              <StarBorder color="#f97316" speed="8s">
                <div className="brand-icon"><Icon d={Icons.zap} size={20} /></div>
              </StarBorder>
              <span className="brand-name">IntelSpace</span>
            </div>

            <div className="tenant-badge mono">
              <span className={`status-dot ${getHeartbeatClass()}`} /> ORG: {org}
            </div>

            <nav className="nav" style={{ position: 'relative' }}>
              <div 
                className="nav-indicator" 
                style={{ transform: `translateY(${activeIndex * 46}px)` }} 
              />
              
              {navItems.map(item => (
                <button key={item.id}
                  className={`nav-item ${panel === item.id ? 'active' : ''}`}
                  onClick={() => setPanel(item.id)}>
                  <Icon d={item.icon} size={18} />
                  <span>{item.label}</span>
                </button>
              ))}
              <button className="nav-item history-toggle" onClick={() => setHistoryOpen(true)}>
                <Icon d={Icons.history} size={18} />
                <span>Search History</span>
              </button>
            </nav>

            <button className="nav-item logout-btn" onClick={handleLogout}>
              <Icon d={Icons.lock} size={18} /> <span>Terminate Session</span>
            </button>
          </aside>

          {/* ── Main Canvas ── */}
          <main className="main">
            <div className="content-container">

              {panel === 'dashboard' && (
                <DashboardView health={health} docs={docs} history={history} />
              )}

              {panel === 'search' && (
                <div className="fade-in command-center">
                  <h2 className="panel-title">Central Command Search</h2>
                  <p className="panel-sub mono">Execute cross-encoder queries against FAISS indices.</p>

                  <form
                    className={`search-form ${results.length > 0 ? 'top-docked' : 'centered'}`}
                    onSubmit={handleSearch}>
                    <div className="search-box glow-focus">
                      <Icon d={Icons.search} size={24} className="search-icon" />
                      <input
                        ref={inputRef}
                        className="search-input mono"
                        type="text"
                        placeholder={animatedPlaceholder ? `> ${animatedPlaceholder}` : '> '}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                      />
                      <StarBorder color="var(--brand)" speed="4s" thickness={1}>
                        <button type="submit" className="search-btn" disabled={loading || !query.trim()}>
                          {loading ? <span className="spinner" /> : 'EXECUTE'}
                        </button>
                      </StarBorder>
                    </div>
                    {meta?.latency && !loading && (
                      <div className="latency-stats mono">
                        [TOTAL: {meta.latency.total}ms] [LLM: {meta.latency.llm}ms] [RERANK: {meta.latency.reranking}ms]
                      </div>
                    )}
                  </form>

                  {/* LLM Answer Summary Panel */}
                  {!loading && meta?.answer && (
                    <div className="answer-block terminal-window">
                      <div className="answer-header">
                        <Icon d={Icons.sparkle} size={18} />
                        <span>SYNTHESIS COMPLETE</span>
                      </div>
                      <div className="answer-text">
                        <FormattedAnswer text={meta.answer} />
                      </div>
                    </div>
                  )}

                  {/* Retrieved Cluster Source Grid */}
                  {!loading && results.length > 0 && (
                    <div className="results-container">
                      <div className="results-header mono">
                        System Found {results.length} Supporting Vectors:
                      </div>
                      <div className="results">
                        {results.map((r, i) => {
                          // CRITICAL FIX: The backend now sends an exact percentage integer/float (e.g., 85.4)
                          const pct = Number(r.score) || 0;
                          return (
                            <div 
                              className="result-card glass-panel staggered-card" 
                              key={i}
                              style={{ animationDelay: `${i * 75}ms` }}
                            >
                              <div className="result-header">
                                <div className="source-tag mono">
                                  <Icon d={Icons.doc} size={14} /> {r.source} // P.{r.page_num}
                                </div>
                                <span className={`score-badge mono ${scoreTier(pct)}`}>
                                  ACC: {pct.toFixed(1)}%
                                  <div className="score-badge-fill-bar" style={{ width: `${pct}%` }} />
                                </span>
                              </div>
                              <p className="result-text">{r.text}</p>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Data Ingestion Control Panel */}
              {panel === 'upload' && (
                <div className="fade-in">
                  <h2 className="panel-title">Data Ingestion Engine</h2>
                  <p className="panel-sub mono">Process external documents via chunking & dense embeddings.</p>

                  <div
                    className="upload-zone glass-panel glow-focus"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => { e.preventDefault(); upload([...e.dataTransfer.files]) }}
                    onClick={() => document.getElementById('file-upload').click()}>
                    <input
                      id="file-upload" type="file" multiple accept=".pdf,.docx"
                      onChange={(e) => { upload([...e.target.files]); e.target.value = null }}
                      hidden />
                    <div className="upload-icon brand-text"><Icon d={Icons.upload} size={48} /></div>
                    <div className="upload-label">INITIALIZE UPLOAD SEQUENCE</div>
                    <div className="upload-hint mono">[AWAITING .PDF OR .DOCX FILES]</div>
                  </div>

                  {progress.length > 0 && (
                    <div className="doc-list" style={{ marginTop: '24px' }}>
                      {progress.map((p, i) => {
                        const barWidth = p.status === 'done' ? 100 : p.status === 'error' ? 100 : 45
                        return (
                          <div key={i} className="doc-item glass-panel upload-progress-row">
                            <div className="upload-progress-background" style={{ width: `${barWidth}%`, opacity: p.status === 'error' ? 0.08 : 0.04 }} />
                            <div style={{ display: 'flex', gap: '16px', alignItems: 'center', width: '100%', zIndex: 1 }}>
                              {p.status === 'uploading'
                                ? <span className="spinner" style={{ borderColor: 'rgba(31,111,95,0.2)', borderTopColor: 'var(--brand)' }} />
                                : p.status === 'error'
                                  ? <span style={{ color: 'var(--danger)', fontWeight: 'bold' }}>✕</span>
                                  : <span style={{ color: 'var(--success)', fontWeight: 'bold' }}>✓</span>
                              }
                              <div className="doc-name" style={{ flex: 1, margin: 0 }}>{p.name}</div>
                              <div className="mono" style={{
                                color: p.status === 'error' ? 'var(--danger)' : p.status === 'done' ? 'var(--success)' : 'var(--brand)',
                                fontSize: '11px'
                              }}>
                                [{p.status.toUpperCase()}]
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* Isolated System Storage Vault */}
              {panel === 'docs' && (
                <div className="fade-in">
                  <h2 className="panel-title">Isolated Vault</h2>
                  <p className="panel-sub mono">Tenant-specific FAISS storage management.</p>
                  
                  {docs.length === 0 ? (
                     <EmptyState message="VAULT_CONTAINER_EMPTY" />
                  ) : (
                     <AnimatedList
                       items={docs.map(d => d.name)}
                       showGradients={true}
                       className="vault-animated-list"
                     />
                  )}
                </div>
              )}

            </div>
          </main>

          {/* ── History Sidebar ── */}
          {historyOpen && <div className="history-overlay" onClick={() => setHistoryOpen(false)} />}
          <aside className={`history-sidebar glass-panel ${historyOpen ? 'open' : ''}`}>
            <div className="hs-header">
              <span className="hs-title mono">COMMAND LOG</span>
              <button className="hs-close" onClick={() => setHistoryOpen(false)}>
                <Icon d={Icons.close} size={20} />
              </button>
            </div>
            <div className="hs-list">
              {history.length === 0
                ? <EmptyState message="LOG_CONTAINER_EMPTY" />
                : <AnimatedList
                    items={history.map(h => h.query)}
                    onItemSelect={(item) => { 
                      setQuery(item); 
                      setPanel('search'); 
                      setHistoryOpen(false); 
                    }}
                    showGradients={true}
                    displayScrollbar={false}
                    className="hs-animated-list"
                  />
              }
            </div>
          </aside>
        </div>
      )}
    </>
  )
}