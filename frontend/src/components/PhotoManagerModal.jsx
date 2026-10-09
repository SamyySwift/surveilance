import { useEffect, useState } from 'react';
import { api } from '../api';
import AuthedImage from './AuthedImage';

export default function PhotoManagerModal({ staff, onClose, onChanged }) {
  const [photos, setPhotos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [addFiles, setAddFiles] = useState([]);
  const [adding, setAdding] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      setPhotos(await api.listPhotos(staff.id));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [staff.id]);

  const removePhoto = async (embeddingId) => {
    try {
      await api.deletePhoto(staff.id, embeddingId);
      await load();
      onChanged();
    } catch (err) {
      setError(err.message);
    }
  };

  const addPhotos = async () => {
    if (addFiles.length === 0) return;
    setAdding(true);
    setError(null);
    try {
      const result = await api.addPhotos(staff.id, addFiles);
      if (result.errors?.length) setError(result.errors.join(' '));
      setAddFiles([]);
      await load();
      onChanged();
    } catch (err) {
      setError(err.message);
    } finally {
      setAdding(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{staff.name}'s photos</h2>
          <button className="btn-icon" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="modal-body">
          {loading ? (
            <p className="empty-state">Loading…</p>
          ) : photos.length === 0 ? (
            <p className="empty-state">No photos enrolled.</p>
          ) : (
            <div className="photo-grid">
              {photos.map((p) => (
                <div key={p.id} className="photo-tile">
                  <AuthedImage path={api.photoUrlPath(staff.id, p.id)} />
                  <button
                    className="photo-remove"
                    onClick={() => removePhoto(p.id)}
                    aria-label="Remove photo"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <label className="field">
            <span>Add more photos</span>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => setAddFiles(Array.from(e.target.files))}
            />
            <span className="field-hint">
              Add a new angle or lighting condition to improve matching, or replace a bad photo by
              deleting it above first.
            </span>
          </label>

          {error && <p className="form-error">{error}</p>}

          <div className="modal-actions">
            <button type="button" className="btn btn-ghost" onClick={onClose}>
              Done
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={addPhotos}
              disabled={adding || addFiles.length === 0}
            >
              {adding ? 'Adding…' : 'Add photos'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
