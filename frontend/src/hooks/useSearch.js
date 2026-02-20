import { useState, useCallback } from 'react'
import { search as apiSearch } from '../utils/api'

export function useSearch() {
  const [results,   setResults]   = useState([])
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)
  const [meta,      setMeta]      = useState(null)
  const [message,   setMessage]   = useState(null)
  const [lastQuery, setLastQuery] = useState('')

  const run = useCallback(async (query, k = 5) => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setMessage(null)
    setLastQuery(query)

    try {
      const { data } = await apiSearch(query, k)
      setResults(data.results || [])
      setMeta({
        latency_ms: data.latency_ms,
        k:          data.k,
        pipeline:   data.pipeline || null,
      })
      setMessage(data.message || null)
    } catch (e) {
      setError(e.response?.data?.error || e.message || 'Search failed')
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [])

  return { results, loading, error, meta, message, lastQuery, run }
}