const BASE_URL = 'http://localhost:5000';

const getHeaders = (token) => {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
};

// OPTIMIZATION: Helper function for safe fetching with a standard timeout
const fetchWithTimeout = async (url, options = {}, timeoutMs = 30000) => {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    if (error.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs / 1000} seconds.`);
    }
    throw error;
  }
};

export const getHealth = async (token) => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/stats`, { 
    headers: getHeaders(token) 
  }, 10000); // 10s timeout for fast endpoints
  
  if (!res.ok) throw res;
  const json = await res.json();
  return { data: json };
};

export const getDocuments = async (token) => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/documents`, { 
    headers: getHeaders(token) 
  }, 15000);
  
  if (!res.ok) throw res;
  const json = await res.json();
  return { data: json };
};

export const deleteDoc = async (docId, token) => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
    headers: getHeaders(token)
  }, 15000);
  
  if (!res.ok) throw res;
  return res.json();
};