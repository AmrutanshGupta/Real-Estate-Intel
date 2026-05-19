const BASE_URL = 'http://localhost:5000';

const getHeaders = (token) => {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
};

export const getHealth = async (token) => {
  // CRITICAL FIX: The backend telemetry endpoint is now /api/stats
  const res = await fetch(`${BASE_URL}/api/stats`, { headers: getHeaders(token) });
  if (!res.ok) throw res;
  
  const json = await res.json();
  // We wrap the response in a 'data' object so App.jsx's `setHealth(h.data)` works automatically
  return { data: json };
};

export const getDocuments = async (token) => {
  // CRITICAL FIX: The backend document vault endpoint is /api/documents
  const res = await fetch(`${BASE_URL}/api/documents`, { headers: getHeaders(token) });
  if (!res.ok) throw res;
  
  const json = await res.json();
  // We wrap the response so App.jsx's `setDocs(d.data.documents)` works automatically
  return { data: json };
};

export const deleteDoc = async (docId, token) => {
  // CRITICAL FIX: The delete endpoint matches the documents endpoint
  const res = await fetch(`${BASE_URL}/api/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
    headers: getHeaders(token)
  });
  if (!res.ok) throw res;
  return res.json();
};