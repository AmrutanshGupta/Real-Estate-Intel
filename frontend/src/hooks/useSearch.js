import { useState, useCallback } from 'react';

export function useSearch() {
  const [results, setResults] = useState([]);
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = useCallback(async (query, k = 10, token = null) => {
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        },
        body: JSON.stringify({ query, k })
      });

      if (!res.ok) {
        if (res.status === 401) throw new Error('Unauthorized');
        throw new Error('Search execution failed');
      }

      const data = await res.json();
      
      setResults(data.results || []);
      setMeta({
        answer: data.answer,
        latency: data.latency,
        query_type: data.query_type,
        entities: data.entities_detected,
        refused: data.refused
      });

      // CRITICAL FIX: Return the fresh payload so App.jsx can read it immediately
      return data; 

    } catch (err) {
      setError(err.message);
      setResults([]);
      setMeta(null);
      
      // Return null on failure so App.jsx knows the search crashed
      return null; 
      
    } finally {
      setLoading(false);
    }
  }, []);

  return { results, loading, error, meta, run };
}