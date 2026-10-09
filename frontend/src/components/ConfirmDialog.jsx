import { useState } from 'react';
import Modal from './Modal';

// In-app replacement for window.confirm / window.prompt. With `noteLabel` it
// also asks for an optional note, passed to onConfirm. onConfirm may return a
// promise: the dialog shows it's working, and if it fails shows the error and
// stays open. The caller closes the dialog once it succeeds.
export default function ConfirmDialog({
  title,
  children,
  confirmLabel,
  busyLabel,
  danger = false,
  noteLabel,
  notePlaceholder,
  onConfirm,
  onCancel,
}) {
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await onConfirm(note.trim());
    } catch (err) {
      setError(err.message || 'Something went wrong. Try again.');
      setBusy(false);
    }
  };

  return (
    <Modal title={title} onClose={onCancel} locked={busy} tone={danger ? 'danger' : undefined}>
      <form className="dialog-form" onSubmit={submit}>
        {children}

        {noteLabel && (
          <label className="field">
            <span>{noteLabel}</span>
            <textarea
              rows={3}
              maxLength={500}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder={notePlaceholder}
              autoFocus
              // Enter adds a line; Cmd/Ctrl+Enter submits.
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) e.currentTarget.form.requestSubmit();
              }}
            />
          </label>
        )}

        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}

        <div className="modal-actions">
          {/* A destructive dialog starts on Cancel, so Enter doesn't delete anything. */}
          <button type="button" className="btn btn-ghost" onClick={onCancel} disabled={busy} autoFocus={danger && !noteLabel}>
            Cancel
          </button>
          <button
            type="submit"
            className={`btn ${danger ? 'btn-destructive' : 'btn-primary'}`}
            disabled={busy}
            autoFocus={!danger && !noteLabel}
          >
            {busy ? busyLabel || 'Working…' : confirmLabel}
          </button>
        </div>
      </form>
    </Modal>
  );
}
