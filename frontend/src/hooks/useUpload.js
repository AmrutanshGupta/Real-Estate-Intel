import { useState } from 'react';

export function useUpload(onSuccess, token = null) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState([]);
  const [error, setError] = useState(null);

  const upload = async (files) => {
    if (!files || files.length === 0) return;
    
    setUploading(true);
    setError(null);
    
    const currentBatch = files.map(f => ({ name: f.name, status: 'uploading', message: null }));
    setProgress(currentBatch);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const formData = new FormData();
      
      // CRITICAL: Matches FastAPI 'files: list[UploadFile]'
      formData.append('files', file); 

      try {
        const res = await fetch('http://localhost:5000/api/upload', {
          method: 'POST',
          headers: {
            ...(token && { 'Authorization': `Bearer ${token}` })
          },
          body: formData
        });

        const data = await res.json();

        if (!res.ok) {
          // Capture HTTP level errors (e.g., 500 Internal Server Error)
          throw new Error(data.detail || `Server error: ${res.status}`);
        }

        // Check if the backend API returned a specific file-level error inside the 200 OK response
        const fileResult = data.files && data.files[0];
        if (fileResult && fileResult.error) {
           throw new Error(fileResult.error);
        }

        setProgress(prev => prev.map((p, idx) => idx === i ? { ...p, status: 'done', message: 'Success' } : p));
        
      } catch (err) {
        console.error(`Upload failed for ${file.name}:`, err);
        // Display the actual error reason to the user on the UI
        setProgress(prev => prev.map((p, idx) => idx === i ? { ...p, status: 'error', message: err.message } : p));
      }
    }
    
    setUploading(false);
    if (onSuccess) onSuccess();
  };

  return { upload, uploading, progress, error };
}