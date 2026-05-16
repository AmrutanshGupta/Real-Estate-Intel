import { useState, useCallback } from 'react'
import { search } from '../utils/api'

/**
 * useSearch — wraps POST /api/search
 *
 * Exposes the full backend response including:
 *   results    → raw retrieved chunks
 *   meta       → answer, sources, refused, query_type, latency, entities_detected
 *   lastQuery  → the last submitted query string
 *   message    → "No matches" message if results empty
 */
export function useSearch() {
  const [results,   setResults]   = useState([])
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)
  const [meta,      setMeta]      = useState(null)
  const [lastQuery, setLastQuery] = useState('')
  const [message,   setMessage]   = useState('')

  const run = useCallback(async (query, k = 5) => {
    if (!query?.trim()) return

    setLoading(true)
    setError(null)
    setMessage('')
    setLastQuery(query.trim())

    try {
      const res  = await search(query.trim(), k)
      const data = res.data

      // Raw retrieval chunks
      setResults(data.results || [])

      // Full response metadata including LLM answer
      setMeta({
        answer:            data.answer      || null,
        sources:           data.sources     || [],
        refused:           data.refused     || false,
        query_type:        data.query_type  || 'general',
        entities_detected: data.entities_detected || [],
        latency:           data.latency     || {},
        cached:            data.cached      || false,
      })

      if ((data.results || []).length === 0) {
        setMessage(data.message || null)
      }

    } catch (err) {
      const msg = err?.response?.data?.detail || err.message || 'Search failed.'
      setError(msg)
      setResults([])
      setMeta(null)
    } finally {
      setLoading(false)
    }
  }, [])

  return { results, loading, error, meta, lastQuery, message, run }
}