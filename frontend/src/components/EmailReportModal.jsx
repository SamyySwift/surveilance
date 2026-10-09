import { useState } from 'react';
import { api } from '../api';
import Modal from './Modal';

// Emails the alarm status now: open alarms, alarms acknowledged in the last
// 24 hours, and whether each camera is online. The server sends it through
// Resend (README "Email alerts"); `settings` is the email part of /api/status.
// New alarms are also emailed automatically, without this dialog.
export default function EmailReportModal({ settings, openCount, onClose }) {
  const [to, setTo] = useState(() => (settings?.defaultTo ?? []).join(', '));
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const [sent, setSent] = useState(null); // { to, subject }
  const configured = settings?.configured === true;

  const send = async (e) => {
    e.preventDefault();
    const recipients = to.split(/[\s,;]+/).filter(Boolean);
    if (recipients.length === 0) {
      setError('Enter at least one email address.');
      return;
    }
    setSending(true);
    setError(null);
    try {
      setSent(await api.emailAlarms(recipients));
    } catch (err) {
      setError(err.message);
    } finally {
      setSending(false);
    }
  };

  return (
    <Modal title="Email alarm status" onClose={onClose} locked={sending}>
      {sent ? (
        <div className="dialog-form">
          <p className="modal-success" role="status">
            Sent to {sent.to.join(', ')}
          </p>
          <p className="modal-text">“{sent.subject}”</p>
          <div className="modal-actions">
            <button type="button" className="btn btn-primary" onClick={onClose} autoFocus>
              Done
            </button>
          </div>
        </div>
      ) : (
        <form className="dialog-form" onSubmit={send}>
          <p className="modal-text">
            Lists the open alarms ({openCount} now), alarms acknowledged in the last 24 hours, and whether each
            camera is online. Snapshots aren't attached, since they show people's faces.
          </p>

          {!configured && (
            <p className="modal-warning">
              Email isn't set up on the server yet. Add <code>RESEND_API_KEY</code> to <code>.env</code> and restart
              the server (README “Email alerts”).
            </p>
          )}

          {settings?.automatic && (
            <p className="form-note">New alarms are emailed automatically to {settings.defaultTo.join(', ')}.</p>
          )}

          <label className="field">
            <span>Send to</span>
            <input
              type="text"
              inputMode="email"
              autoComplete="email"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              placeholder="name@example.com"
              autoFocus
            />
            <span className="field-hint">Separate several addresses with commas.</span>
          </label>

          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}

          <div className="modal-actions">
            <button type="button" className="btn btn-ghost" onClick={onClose} disabled={sending}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={!configured || sending}>
              {sending ? 'Sending…' : 'Send'}
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
}
