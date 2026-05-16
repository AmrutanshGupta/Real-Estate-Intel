import { useState, useEffect, useRef, useCallback } from 'react'
import { getHealth, getDocuments, deleteDoc } from './utils/api'
import { useSearch } from './hooks/useSearch'
import { useUpload } from './hooks/useUpload'
import './App.css'

// ── Icons ──────────────────────────────────────────────────────────────
const Icon = ({ d, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
  zap:       "M13 2L3 14h9l-1 8 10-12h-9l1-8z"
}

// ── Markdown Formatting Helper ─────────────────────────────────────────
const FormattedAnswer = ({ text }) => {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    // 1. Preserve empty lines as spacing
    if (!trimmed) {
      elements.push(<div key={`space-${index}`} style={{ height: '12px' }} />);
      return;
    }

    // 2. Helper to make **text** actually bold
    const parseBold = (str) => {
      const parts = str.split(/(\*\*.*?\*\*)/g);
      return parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i} style={{ color: 'var(--text-main)', fontWeight: 700 }}>{part.slice(2, -2)}</strong>;
        }
        return part;
      });
    };

    // 3. Handle Lists (Bullets "- " or Numbers "1. ")
    if (trimmed.startsWith('- ') || /^\d+\.\s/.test(trimmed)) {
      const isBullet = trimmed.startsWith('- ');
      const content = isBullet ? trimmed.slice(2) : trimmed.replace(/^\d+\.\s/, '');
      const prefix = isBullet ? '•' : trimmed.match(/^\d+\./)[0];

      elements.push(
        <div key={index} style={{ display: 'flex', gap: '10px', marginBottom: '6px', paddingLeft: '16px' }}>
          <span style={{ color: 'var(--brand)', fontWeight: 'bold' }}>{prefix}</span>
          <span style={{ flex: 1 }}>{parseBold(content)}</span>
        </div>
      );
    } 
    // 4. Handle regular paragraphs
    else {
      elements.push(
        <div key={index} style={{ marginBottom: '8px', lineHeight: '1.7' }}>
          {parseBold(trimmed)}
        </div>
      );
    }
  });

  return <div className="formatted-content">{elements}</div>;
};

// ── Dashboard Component ────────────────────────────────────────────────
function DashboardView({ health, docs, history, onNavigate }) {
  // Calculate total chunks dynamically from the docs array
  const totalChunks = docs.reduce((acc, d) => acc + d.chunks, 0);
  
  // Use health.vectors if provided by API, otherwise fallback to the calculated chunks
  const totalVectors = health?.vectors || totalChunks;
  
  return (
    <div>
      <h2 className="panel-title">System Dashboard</h2>
      <p className="panel-sub">Overview of your real estate intelligence infrastructure.</p>
      
      <div className="dash-grid">
        <div className="stat-card">
          <span className="stat-label">Total Documents</span>
          <span className="stat-value">{docs.length}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Indexed Vectors</span>
          <span className="stat-value brand">{totalVectors.toLocaleString()}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Processed Chunks</span>
          <span className="stat-value">{totalChunks.toLocaleString()}</span>
        </div>
      </div>

      <div className="dash-sections">
        <div className="dash-card">
          <h3 className="card-header">Recent Queries</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {history.slice(0, 4).map((h, i) => (
              <div key={i} style={{ borderBottom: '1px solid var(--border)', paddingBottom: '8px' }}>
                <div style={{ fontWeight: '600', color: 'var(--text-main)' }}>"{h.query}"</div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Found {h.results} results</div>
              </div>
            ))}
            {history.length === 0 && <p style={{ color: 'var(--text-muted)' }}>No queries run yet.</p>}
          </div>
        </div>

        <div className="dash-card">
          <h3 className="card-header">Recent Documents</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {docs.slice(0, 4).map((d, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Icon d={Icons.doc} size={16} />
                <div>
                  <div style={{ fontWeight: '600', color: 'var(--text-main)' }}>{d.name}</div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{d.chunks} chunks</div>
                </div>
              </div>
            ))}
            {docs.length === 0 && <p style={{ color: 'var(--text-muted)' }}>No documents uploaded.</p>}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────
export default function App() {
  const [panel, setPanel] = useState('dashboard')
  const [query, setQuery] = useState('')
  const [history, setHistory] = useState([])
  const [historyOpen, setHistoryOpen] = useState(false)
  const inputRef = useRef()

  const [health, setHealth] = useState(null)
  const [docs, setDocs] = useState([])

  const { results, loading, error: searchError, meta, run } = useSearch()
  
  const refreshAll = useCallback(async () => {
    try {
      const [h, d] = await Promise.all([getHealth(), getDocuments()])
      setHealth(h.data)
      setDocs(d.data.documents || [])
    } catch {}
  }, [])

  const { upload, uploading, progress, error: uploadError } = useUpload(refreshAll)

  // Initialization & History Persistence
  useEffect(() => { refreshAll() }, [refreshAll])
  useEffect(() => {
    const saved = localStorage.getItem('rei_history')
    if (saved) try { setHistory(JSON.parse(saved)) } catch {}
  }, [])
  useEffect(() => { localStorage.setItem('rei_history', JSON.stringify(history)) }, [history])

  const handleSearch = (e) => {
    e?.preventDefault()
    if (!query.trim()) return
    run(query.trim(), 5)
    
    // Save to history
    const entry = { query: query.trim(), results: results?.length || 0, ts: Date.now() }
    setHistory(prev => [entry, ...prev.filter(h => h.query !== query.trim())].slice(0, 20))
  }

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Icons.dashboard },
    { id: 'search', label: 'Query Interface', icon: Icons.search },
    { id: 'upload', label: 'Data Ingestion', icon: Icons.upload },
    { id: 'docs', label: `Datastore (${docs.length})`, icon: Icons.doc },
  ]

    const totalChunks = docs.reduce((acc, d) => acc + d.chunks, 0);
  const displayVectors = health?.vectors || totalChunks;
  
  return (
    <div className="layout">
      {/* ── Left Sidebar ── */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Icon d={Icons.zap} size={20} /></div>
          <span className="brand-name">IntelSpace</span>
        </div>

        <nav className="nav">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`nav-item ${panel === item.id ? 'active' : ''}`}
              onClick={() => setPanel(item.id)}
            >
              <Icon d={item.icon} size={18} />
              <span>{item.label}</span>
            </button>
          ))}
          
          <button className="nav-item history-toggle" onClick={() => setHistoryOpen(true)}>
            <Icon d={Icons.history} size={18} />
            <span>Search History</span>
          </button>
        </nav>
      </aside>

      {/* ── Main Workspace ── */}
      <main className="main">
        <div className="content-container">

          {/* DASHBOARD PANEL */}
          {panel === 'dashboard' && (
            <DashboardView health={health} docs={docs} history={history} onNavigate={setPanel} />
          )}

          {/* SEARCH PANEL */}
          {panel === 'search' && (
            <div>
              <h2 className="panel-title">Query Interface</h2>
              <p className="panel-sub">Search your real estate corpus and generate AI insights.</p>
              
              <form className="search-form" onSubmit={handleSearch}>
                <div className="search-box">
                  <Icon d={Icons.search} />
                  <input
                    ref={inputRef}
                    className="search-input"
                    type="text"
                    placeholder="Ask about landmarks, project amenities, or total carpet area..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  <button type="submit" className="search-btn" disabled={loading || !query.trim()}>
                    {loading ? <span className="spinner" /> : 'Search Docs'}
                  </button>
                </div>
              </form>

                {/* LLM Answer Block */}
                {!loading && meta?.answer && (
                  <div className="answer-block">
                    <div className="answer-header">
                      <Icon d={Icons.sparkle} size={18} />
                      <span>AI Synthesis</span>
                    </div>
                    <div className="answer-text">
                      <FormattedAnswer text={meta.answer} />
                    </div>
                  </div>
                )}

              {/* Retrieved Sources */}
              {!loading && results.length > 0 && (
                <div>
                  <div className="results-header">Retrieved Source Chunks</div>
                  <div className="results">
                    {results.map((r, i) => (
                      <div className="result-card" key={i}>
                        <div className="result-header">
                          <div className="source-tag">
                            <Icon d={Icons.doc} size={14} /> {r.source} (Page {r.page_num})
                          </div>
                          <span className="score-badge">Match: {(r.score * 100).toFixed(1)}%</span>
                        </div>
                        <p className="result-text">{r.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* UPLOAD PANEL */}
          {panel === 'upload' && (
            <div>
              <h2 className="panel-title">Data Ingestion</h2>
              <p className="panel-sub">Upload PDF or DOCX files to safely extract and embed their text.</p>
              
              <div 
                className="upload-zone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => { e.preventDefault(); upload([...e.dataTransfer.files]) }}
                onClick={() => document.getElementById('file-upload').click()}
              >
                <input id="file-upload" type="file" multiple accept=".pdf,.docx" onChange={(e) => upload([...e.target.files])} hidden />
                <div className="upload-icon"><Icon d={Icons.upload} size={36} /></div>
                <div className="upload-label">Drag & Drop files here or browse</div>
                <div className="upload-hint">Supported formats: PDF, DOCX (Max 50MB)</div>
              </div>

              {progress.length > 0 && (
                <div className="doc-list">
                  {progress.map((p, i) => (
                    <div key={i} className="doc-item">
                      <Icon d={Icons.doc} size={20} />
                      <div className="doc-name" style={{flex: 1}}>{p.name}</div>
                      <div style={{ fontWeight: '600', color: p.status === 'error' ? 'var(--danger)' : 'var(--success)' }}>
                        {p.status.toUpperCase()}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {/* DATABASE PANEL */}
          {panel === 'docs' && (
            <div>
              <h2 className="panel-title">Vector Datastore</h2>
              <p className="panel-sub">Manage the {docs.length} documents currently indexed in your system.</p>
              
              <div className="doc-list">
                {docs.map((d, i) => (
                  <div key={i} className="doc-item">
                    <Icon d={Icons.doc} size={24} />
                    <div>
                      <div className="doc-name">{d.name}</div>
                      <div className="doc-meta">{d.pages} pages • {d.chunks} text chunks</div>
                    </div>
                    <button className="doc-delete" onClick={() => { if(window.confirm(`Delete ${d.name}?`)) { deleteDoc(d.name); refreshAll() } }}>
                      <Icon d={Icons.trash} size={16} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </main>

      {/* ── Slide-out History Sidebar ── */}
      {historyOpen && <div className="history-overlay" onClick={() => setHistoryOpen(false)} />}
      <aside className={`history-sidebar ${historyOpen ? 'open' : ''}`}>
        <div className="hs-header">
          <span className="hs-title">Query History</span>
          <button className="hs-close" onClick={() => setHistoryOpen(false)}><Icon d={Icons.close} size={20} /></button>
        </div>
        <div className="hs-list">
          {history.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: '20px' }}>No history found.</p>
          ) : (
            history.map((h, i) => (
              <div key={i} className="hs-item" onClick={() => { setQuery(h.query); setPanel('search'); setHistoryOpen(false); }}>
                <div className="hs-query">{h.query}</div>
                <div className="hs-meta">{h.results} results</div>
              </div>
            ))
          )}
        </div>
      </aside>

    </div>
  )
}