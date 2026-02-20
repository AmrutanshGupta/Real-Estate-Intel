import { useState, useEffect, useRef, useCallback } from 'react'
import { getHealth, getDocuments, deleteDoc } from './utils/api'
import { useSearch } from './hooks/useSearch'
import { useUpload } from './hooks/useUpload'
import './App.css'

// ── Icons ─────────────────────────────────────────────────────────────────────
const Icon = ({ d, size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
)

const Icons = {
  search:   "M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z",
  upload:   "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12",
  doc:      "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6",
  trash:    "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6",
  close:    "M18 6L6 18M6 6l12 12",
  check:    "M20 6L9 17l-5-5",
  warning:  "M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z",
  building: "M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM9 22V12h6v10",
  zap:      "M13 2L3 14h9l-1 8 10-12h-9l1-8z",
  file:     "M13 2H6a2 2 0 0 0-2 2v16c0 1.1.9 2 2 2h12a2 2 0 0 0 2-2V9l-7-7z",
  info:     "M12 16v-4M12 8h.01M22 12A10 10 0 1 1 2 12a10 10 0 0 1 20 0z",
  history:  "M12 8v4l3 3M3.05 11a9 9 0 1 0 .5-3",
}

// ── StatusDot ─────────────────────────────────────────────────────────────────
function StatusDot({ ready }) {
  return (
    <span className={`status-dot ${ready ? 'ready' : 'offline'}`}>
      <span className="dot-pulse" />
    </span>
  )
}

// ── ScoreBadge ────────────────────────────────────────────────────────────────
function ScoreBadge({ score }) {
  const pct = Math.round(score * 100)
  const cls = pct >= 70 ? 'high' : pct >= 45 ? 'mid' : 'low'
  return <span className={`score-badge ${cls}`}>{pct}% Match</span>
}

// ── PipelineBadges ────────────────────────────────────────────────────────────
function PipelineBadges({ pipeline }) {
  if (!pipeline) return null
  return (
    <div className="pipeline-badges">
      {pipeline.expanded && <span className="pip-badge expanded">Expanded</span>}
      {pipeline.hybrid   && <span className="pip-badge hybrid">Hybrid</span>}
      {pipeline.reranked && <span className="pip-badge reranked">Reranked</span>}
    </div>
  )
}

// ── UploadZone ────────────────────────────────────────────────────────────────
function UploadZone({ onFiles }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const files = [...e.dataTransfer.files].filter(f => f.name.endsWith('.pdf'))
    if (files.length) onFiles(files)
  }, [onFiles])

  const handleChange = (e) => {
    const files = [...e.target.files]
    if (files.length) onFiles(files)
    e.target.value = ''
  }

  return (
    <div
      className={`upload-zone ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" multiple accept=".pdf" onChange={handleChange} hidden />
      <div className="upload-icon"><Icon d={Icons.upload} size={22} /></div>
      <p className="upload-label">Drop PDFs here or <span>browse</span></p>
      <p className="upload-hint">Max 50 MB per file</p>
    </div>
  )
}

// ── ProgressList ──────────────────────────────────────────────────────────────
function ProgressList({ items }) {
  if (!items.length) return null
  return (
    <div className="progress-list">
      {items.map((f, i) => (
        <div key={i} className={`progress-item ${f.status}`}>
          <Icon d={Icons.file} size={14} />
          <span className="prog-name">{f.name}</span>
          {f.status === 'uploading' && <span className="prog-tag uploading">Processing…</span>}
          {f.status === 'done'      && <span className="prog-tag done">{f.chunks} chunks</span>}
          {f.status === 'error'     && <span className="prog-tag error">{f.error || 'Failed'}</span>}
        </div>
      ))}
    </div>
  )
}

// ── DocumentList ──────────────────────────────────────────────────────────────
function DocumentList({ docs, onDelete }) {
  if (!docs.length) return (
    <div className="empty-docs">
      <Icon d={Icons.building} size={32} />
      <p>No documents ingested yet.</p>
      <p>Upload PDFs above to get started.</p>
    </div>
  )
  return (
    <div className="doc-list">
      {docs.map((d, i) => (
        <div key={i} className="doc-item">
          <div className="doc-icon"><Icon d={Icons.doc} size={16} /></div>
          <div className="doc-info">
            <span className="doc-name" title={d.name}>{d.name}</span>
            <span className="doc-meta">{d.pages}p · {d.chunks} chunks</span>
          </div>
          <button className="doc-delete" onClick={() => onDelete(d.name)} title="Remove">
            <Icon d={Icons.trash} size={14} />
          </button>
        </div>
      ))}
    </div>
  )
}

// ── ResultCard ────────────────────────────────────────────────────────────────
function ResultCard({ result, index, query }) {
  const [expanded, setExpanded] = useState(false)
  const text    = result.text || ''
  const preview = text.length > 280 ? text.slice(0, 280) + '…' : text

  // Highlight the best matching sentence + keyword terms
  const renderText = (str) => {
    const sentences = str.split(/(?<=[.!?])\s+/)
    const words     = query.toLowerCase().split(/\s+/).filter(w => w.length > 3)

    // Find best sentence by keyword overlap
    const scored = sentences.map(s => ({
      text:  s,
      score: words.filter(w => s.toLowerCase().includes(w)).length,
    }))
    const best = scored.sort((a, b) => b.score - a.score)[0]

    return sentences.map((s, i) => {
      const isBest = s === best?.text && best.score > 0

      // Highlight individual keywords within each sentence
      const parts = words.length
        ? s.split(new RegExp(`(${words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi'))
        : [s]

      const highlighted = parts.map((p, j) =>
        words.some(w => p.toLowerCase() === w)
          ? <mark key={j}>{p}</mark>
          : p
      )

      return isBest
        ? <span key={i} className="best-sentence">{highlighted} </span>
        : <span key={i}>{highlighted} </span>
    })
  }

  return (
    <div className="result-card" style={{ animationDelay: `${index * 60}ms` }}>
      <div className="result-header">
        <div className="result-rank">#{index + 1}</div>
        <div className="result-source">
          <Icon d={Icons.doc} size={13} />
          <span className="source-name">{result.source}</span>
          <span className="source-page">p.{result.page_num}</span>
        </div>
        <ScoreBadge score={result.score} />
      </div>
      <div className="result-body">
        <p className="result-text">
          {renderText(expanded ? text : preview)}
        </p>
        {text.length > 280 && (
          <button className="expand-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Show less' : 'Show more'}
          </button>
        )}
      </div>
    </div>
  )
}

// ── NoResults ─────────────────────────────────────────────────────────────────
function NoResults({ query, message, onSelect }) {
  const rephrases = [
    "distance from metro station",
    "price per square feet",
    "car parking available",
    "possession handover date",
    "developer builder name",
  ]
  return (
    <div className="empty-results">
      <div className="empty-icon"><Icon d={Icons.search} size={36} /></div>
      <p className="empty-title">
        {message || `No confident matches for "${query}"`}
      </p>
      <p className="empty-sub">
        The query may use words not present in your documents.
      </p>
      <div className="rephrase-box">
        <p className="rephrase-label">Try instead:</p>
        <div className="suggest-chips">
          {rephrases.map((s, i) => (
            <button key={i} className="chip" onClick={() => onSelect(s)}>{s}</button>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── SuggestedQueries ──────────────────────────────────────────────────────────
function SuggestedQueries({ onSelect }) {
  const suggestions = [
    "What are the nearby landmarks?",
    "What is the total carpet area?",
    "What amenities does the project offer?",
    "What is the possession date?",
    "Who is the developer?",
    "What is the RERA number?",
    "Is parking available?",
    "What schools are nearby?",
  ]
  return (
    <div className="suggestions">
      <p className="suggest-label">Try a query</p>
      <div className="suggest-chips">
        {suggestions.map((s, i) => (
          <button key={i} className="chip" onClick={() => onSelect(s)}>{s}</button>
        ))}
      </div>
    </div>
  )
}

// ── SearchHistory ─────────────────────────────────────────────────────────────
function SearchHistory({ history, onSelect, onClear }) {
  if (!history.length) return null
  return (
    <div className="search-history">
      <div className="history-header">
        <span className="suggest-label">Recent</span>
        <button className="history-clear" onClick={onClear}>Clear</button>
      </div>
      <div className="suggest-chips">
        {history.map((q, i) => (
          <button key={i} className="chip history-chip" onClick={() => onSelect(q)}>
            <Icon d={Icons.history} size={11} />
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [health,  setHealth]  = useState(null)
  const [docs,    setDocs]    = useState([])
  const [query,   setQuery]   = useState('')
  const [k,       setK]       = useState(5)
  const [panel,   setPanel]   = useState('search')
  const [history, setHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const inputRef = useRef()

  const { results, loading, error, meta, lastQuery, message, run } = useSearch()

  const refreshAll = useCallback(async () => {
    try {
      const [h, d] = await Promise.all([getHealth(), getDocuments()])
      setHealth(h.data)
      setDocs(d.data.documents || [])
    } catch {}
  }, [])

  const { upload, uploading, progress } = useUpload(refreshAll)

  useEffect(() => { refreshAll() }, [refreshAll])

  const handleSearch = (e) => {
    e.preventDefault()
    const q = query.trim()
    if (!q) return
    run(q, k)
    // Add to history (max 8, no duplicates)
    setHistory(prev => [q, ...prev.filter(h => h !== q)].slice(0, 8))
    setShowHistory(false)
  }

  const handleDelete = async (name) => {
    if (!confirm(`Remove "${name}" from the index?`)) return
    try { await deleteDoc(name); await refreshAll() } catch {}
  }

  const handleSuggest = (s) => {
    setQuery(s)
    setPanel('search')
    run(s, k)
    setHistory(prev => [s, ...prev.filter(h => h !== s)].slice(0, 8))
    setShowHistory(false)
    setTimeout(() => inputRef.current?.focus(), 50)
  }

  return (
    <div className="layout">

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Icon d={Icons.building} size={18} /></div>
          <div>
            <h1 className="brand-name">Real Estate<br /><em>Intel</em></h1>
          </div>
        </div>

        <div className="health-card">
          <StatusDot ready={health?.ready} />
          <div className="health-info">
            <span className="health-status">{health?.ready ? 'Index Ready' : 'No Index'}</span>
            <span className="health-meta">
              {health?.vectors ? `${health.vectors.toLocaleString()} vectors` : 'Upload to start'}
            </span>
          </div>
        </div>

        {/* Pipeline status */}
        {health?.ready && (
          <div className="pipeline-status">
            <div className={`pip-status-row ${health.bm25 ? 'on' : 'off'}`}>
              <span className="pip-dot" />
              <span>Hybrid BM25</span>
            </div>
            <div className="pip-status-row on">
              <span className="pip-dot" />
              <span>Semantic FAISS</span>
            </div>
            <div className="pip-status-row on">
              <span className="pip-dot" />
              <span>Cross-encoder Rerank</span>
            </div>
          </div>
        )}

        <nav className="nav">
          {[
            { id: 'search', label: 'Search',   icon: Icons.search },
            { id: 'upload', label: 'Upload',   icon: Icons.upload },
            { id: 'docs',   label: `Documents${docs.length ? ` (${docs.length})` : ''}`, icon: Icons.doc },
          ].map(item => (
            <button
              key={item.id}
              className={`nav-item ${panel === item.id ? 'active' : ''}`}
              onClick={() => setPanel(item.id)}
            >
              <Icon d={item.icon} size={16} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="mono text-dim">v{health?.version || '1.0.0'}</span>
          <span className="sidebar-model">{health?.model || 'MiniLM-L6'}</span>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main">

        {/* ── SEARCH PANEL ── */}
        {panel === 'search' && (
          <div className="panel-search">
            <div className="search-header">
              <h2 className="panel-title serif">Query your documents</h2>
              <p className="panel-sub">Ask anything about your real estate corpus</p>
            </div>

            <form className="search-form" onSubmit={handleSearch}>
              <div className="search-box">
                <span className="search-icon"><Icon d={Icons.search} size={18} /></span>
                <input
                  ref={inputRef}
                  className="search-input"
                  type="text"
                  placeholder="What are the nearby landmarks?"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onFocus={() => setShowHistory(true)}
                  onBlur={() => setTimeout(() => setShowHistory(false), 150)}
                  autoFocus
                />
                <div className="search-controls">
                  <div className="k-select">
                    <label>Top</label>
                    <select value={k} onChange={e => setK(Number(e.target.value))}>
                      {[3, 5, 8, 10, 15, 20].map(n => <option key={n}>{n}</option>)}
                    </select>
                  </div>
                  <button
                    type="submit"
                    className="search-btn"
                    disabled={loading || !query.trim()}
                  >
                    {loading ? <span className="spinner" /> : <Icon d={Icons.zap} size={16} />}
                    {loading ? 'Searching…' : 'Search'}
                  </button>
                </div>
              </div>

              {/* Search history dropdown */}
              {showHistory && history.length > 0 && (
                <div className="history-dropdown">
                  {history.map((q, i) => (
                    <button
                      key={i}
                      type="button"
                      className="history-item"
                      onClick={() => handleSuggest(q)}
                    >
                      <Icon d={Icons.history} size={13} />
                      <span>{q}</span>
                    </button>
                  ))}
                </div>
              )}
            </form>

            {error && (
              <div className="alert error">
                <Icon d={Icons.warning} size={16} />
                {error}
              </div>
            )}

            {/* Show pipeline badges after search */}
            {meta && (
              <div className="result-meta">
                <span>{results.length} result{results.length !== 1 ? 's' : ''}</span>
                <span className="sep">·</span>
                <PipelineBadges pipeline={meta.pipeline} />
              </div>
            )}

            {/* Suggestions when no query yet */}
            {!lastQuery && !loading && (
              <>
                <SuggestedQueries onSelect={handleSuggest} />
                {history.length > 0 && (
                  <SearchHistory
                    history={history}
                    onSelect={handleSuggest}
                    onClear={() => setHistory([])}
                  />
                )}
              </>
            )}

            {/* Loading skeleton */}
            {loading && (
              <div className="loading-state">
                <div className="skeleton-row" />
                <div className="skeleton-row" style={{ width: '85%' }} />
                <div className="skeleton-row" style={{ width: '70%' }} />
              </div>
            )}

            {/* Results */}
            {!loading && results.length > 0 && (
              <div className="results">
                {results.map((r, i) => (
                  <ResultCard key={i} result={r} index={i} query={lastQuery} />
                ))}
              </div>
            )}

            {/* No results — helpful message instead of blank */}
            {!loading && lastQuery && results.length === 0 && !error && (
              <NoResults
                query={lastQuery}
                message={message}
                onSelect={handleSuggest}
              />
            )}
          </div>
        )}

        {/* ── UPLOAD PANEL ── */}
        {panel === 'upload' && (
          <div className="panel-upload">
            <div className="panel-header">
              <h2 className="panel-title serif">Upload Documents</h2>
              <p className="panel-sub">Add PDFs to your knowledge base</p>
            </div>

            <UploadZone onFiles={upload} />

            {uploading && (
              <div className="alert info">
                <span className="spinner small" />
                Extracting text and embedding…
              </div>
            )}

            <ProgressList items={progress} />

            {progress.some(p => p.status === 'done') && (
              <div className="alert success">
                <Icon d={Icons.check} size={16} />
                Documents added to index. Switch to Search to query them.
              </div>
            )}

            <div className="upload-tips">
              <h3>What happens when you upload?</h3>
              <ol>
                <li>PyMuPDF tries 4 extraction strategies, picks most readable</li>
                <li>Text is cleaned — removes font artifacts and garbled chars</li>
                <li>Split into sentence-aware chunks (~250 tokens, 2-sentence overlap)</li>
                <li>Each chunk embedded with <code>all-mpnet-base-v2</code></li>
                <li>Embeddings L2-normalised → added to FAISS + BM25 index</li>
                <li>Both indexes persisted to disk for restart durability</li>
              </ol>
            </div>
          </div>
        )}

        {/* ── DOCUMENTS PANEL ── */}
        {panel === 'docs' && (
          <div className="panel-docs">
            <div className="panel-header-row">
              <div>
                <h2 className="panel-title serif">Documents</h2>
                <p className="panel-sub">{docs.length} file{docs.length !== 1 ? 's' : ''} in index</p>
              </div>
              {health && (
                <div className="index-stat-row">
                  <div className="idx-stat">
                    <span className="idx-num">{health.vectors?.toLocaleString() || 0}</span>
                    <span className="idx-label">vectors</span>
                  </div>
                  <div className="idx-stat">
                    <span className="idx-num">{docs.length}</span>
                    <span className="idx-label">documents</span>
                  </div>
                </div>
              )}
            </div>
            <DocumentList docs={docs} onDelete={handleDelete} />
          </div>
        )}
      </main>
    </div>
  )
}