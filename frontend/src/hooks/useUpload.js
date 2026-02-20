import { useState, useCallback } from 'react'
import { uploadPDFs } from '../utils/api'

export function useUpload(onComplete) {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress]  = useState([])
  const [error, setError]        = useState(null)

  const upload = useCallback(async (files) => {
    if (!files.length) return
    setUploading(true)
    setError(null)
    setProgress(files.map(f => ({ name: f.name, status: 'uploading' })))

    const form = new FormData()
    files.forEach(f => form.append('files', f))

    try {
      const { data } = await uploadPDFs(form)
      setProgress(
        (data.files || []).map(f => ({
          name:   f.file,
          status: f.error ? 'error' : 'done',
          chunks: f.chunks,
          pages:  f.pages,
          ms:     f.ingest_ms,
          error:  f.error,
        }))
      )
      onComplete?.(data)
    } catch (e) {
      setError(e.response?.data?.error || e.message || 'Upload failed')
      setProgress(prev => prev.map(p => ({ ...p, status: 'error' })))
    } finally {
      setUploading(false)
    }
  }, [onComplete])

  return { upload, uploading, progress, error }
}
