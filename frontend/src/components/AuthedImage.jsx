import { useEffect, useState } from 'react';
import { api } from '../api';

// An <img> for an image behind the API token (enrollment photos, alarm
// snapshots). Fetches it with the auth header and shows it from an object
// URL, which is released when the image changes or unmounts.
export default function AuthedImage({ path, alt = '', className, onClick }) {
  const [src, setSrc] = useState(null);

  useEffect(() => {
    let cancelled = false;
    let url = null;

    api
      .objectUrl(path)
      .then((u) => {
        if (cancelled) URL.revokeObjectURL(u);
        else setSrc((url = u));
      })
      .catch((err) => console.error('Failed to load image:', err));

    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [path]);

  if (!src) return <span className={`img-placeholder ${className || ''}`} aria-hidden="true" />;
  return <img src={src} alt={alt} className={className} onClick={onClick} />;
}
