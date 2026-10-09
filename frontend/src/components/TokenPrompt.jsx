import { useState } from 'react';
import { api, setToken } from '../api';

export default function TokenPrompt({ notice, onAuthenticated }) {
  const [value, setValue] = useState('');
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    const candidate = value.trim();
    if (!candidate) {
      setError('Enter the API token.');
      return;
    }

    setChecking(true);
    setError(null);
    setToken(candidate);
    try {
      await api.status();
      onAuthenticated();
    } catch (err) {
      setToken(null);
      setError(
        err.status === 401
          ? 'That token was rejected.'
          : `Couldn't reach the backend at ${api.base}: ${err.message}`
      );
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="modal-backdrop">
      <div className="modal" role="dialog" aria-labelledby="token-prompt-title">
        <div className="modal-header">
          <h2 id="token-prompt-title">Gatekeeper sign-in</h2>
        </div>

        <form onSubmit={submit} className="modal-body">
          {notice && <p className="form-error">{notice}</p>}

          <label className="field">
            <span>API token</span>
            <input
              type="password"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              autoComplete="current-password"
              autoFocus
            />
            <span className="field-hint">
              The backend prints this at startup, or uses GATEKEEPER_API_TOKEN if you set it.
            </span>
          </label>

          {error && <p className="form-error">{error}</p>}

          <div className="modal-actions">
            <button type="submit" className="btn btn-primary" disabled={checking}>
              {checking ? 'Checking…' : 'Sign in'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
