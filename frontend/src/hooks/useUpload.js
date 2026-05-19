import { useState } from 'react';

export function useUpload(onSuccess, token = null) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState([]);
  const [error, setError] = useState(null);

  const upload = async (files) => {
    if (!files || files.length === 0) return;
    
    setUploading(true);
    setError(null);
    
    const currentBatch = files.map(f => ({ name: f.name, status: 'uploading' }));
    setProgress(currentBatch);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const formData = new FormData();
      
      // CRITICAL FIX: This must be 'files' (plural) to match FastAPI
      formData.append('files', file); 

      try {
        const res = await fetch('http://localhost:5000/api/upload', {
          method: 'POST',
          headers: {
            ...(token && { 'Authorization': `Bearer ${token}` })
          },
          body: formData
        });

        if (!res.ok) throw new Error(`Ingestion failed for ${file.name}`);

        setProgress(prev => prev.map((p, idx) => idx === i ? { ...p, status: 'done' } : p));
        
      } catch (err) {
        setProgress(prev => prev.map((p, idx) => idx === i ? { ...p, status: 'error' } : p));
      }
    }
    
    setUploading(false);
    if (onSuccess) onSuccess();
  };

  return { upload, uploading, progress, error };
}