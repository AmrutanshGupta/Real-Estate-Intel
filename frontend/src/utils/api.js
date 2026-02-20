import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 30000 })

export const getHealth    = ()          => api.get('/health')
export const getDocuments = ()          => api.get('/documents')
export const uploadPDFs   = (formData)  => api.post('/upload', formData, {
  headers: { 'Content-Type': 'multipart/form-data' },
  timeout: 120000,
})
export const search       = (query, k = 5) => api.post('/search', { query, k })
export const deleteDoc    = (name)      => api.delete(`/documents/${encodeURIComponent(name)}`)
