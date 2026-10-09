import { useEffect, useId } from 'react';
import { createPortal } from 'react-dom';

// The dashboard's modal shell: backdrop, title bar and close button. Escape
// or a click outside closes it, except while `locked` (e.g. mid-request).
// Rendered into <body>: panels use backdrop-filter, which would otherwise
// trap the fixed-position backdrop inside whichever panel opened it.
export default function Modal({ title, onClose, locked = false, tone, children }) {
  const titleId = useId();

  useEffect(() => {
    if (locked) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [locked, onClose]);

  return createPortal(
    // mousedown, not click: a drag that starts in a text field and ends
    // outside the dialog shouldn't close it and lose what was typed.
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !locked) onClose();
      }}
    >
      <div className={`modal ${tone ? `modal--${tone}` : ''}`} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="modal-header">
          <h2 id={titleId}>{title}</h2>
          <button className="btn-icon" onClick={onClose} disabled={locked} aria-label="Close">
            ×
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>,
    document.body
  );
}
