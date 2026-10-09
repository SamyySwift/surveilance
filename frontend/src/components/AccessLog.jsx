import { useState } from 'react';
import { api } from '../api';
import { formatTime } from '../format';
import AuthedImage from './AuthedImage';

// The audit trail of badge + face decisions: who badged where, what the
// system decided and why, with the frame it decided on.
export default function AccessLog({ events }) {
  const [openSnapshot, setOpenSnapshot] = useState(null);

  return (
    <div className="panel access-panel">
      <div className="panel-header">
        <h2>Badge decisions</h2>
        <span className="panel-header-meta">latest {events.length}</span>
      </div>

      {events.length === 0 ? (
        <p className="empty-state">No badge checks yet. Each badge read at a door appears here with the decision and the frame it was based on.</p>
      ) : (
        <ul className="access-list">
          {events.map((e) => (
            <li key={e.id} className="access-row">
              <span className={`decision decision--${e.decision}`}>
                <span aria-hidden="true">{e.decision === 'grant' ? '✓' : '✕'}</span>
                {e.decision === 'grant' ? 'Granted' : 'Denied'}
              </span>
              <div className="access-body">
                <div className="access-who">
                  {e.name || 'Unknown badge'}
                  <span className="access-badge">badge {e.badgeId || '-'}</span>
                </div>
                <div className="alarm-meta">
                  {formatTime(e.time)} · {e.zone} · {e.reason}
                  {e.matchScore != null && ` · match ${e.matchScore.toFixed(2)}`}
                  {e.livenessScore != null && ` · liveness ${e.livenessScore.toFixed(2)}`}
                </div>
                {openSnapshot === e.id && e.snapshot && (
                  <AuthedImage className="snapshot-large" path={api.snapshotPath(e.snapshot)} alt="Decision snapshot" />
                )}
              </div>
              <div className="row-actions row-actions--thumb-only">
                <span className="thumb-slot">
                  {e.snapshot && (
                    <button
                      className="snapshot-thumb-btn"
                      onClick={() => setOpenSnapshot(openSnapshot === e.id ? null : e.id)}
                      aria-label={openSnapshot === e.id ? 'Hide snapshot' : 'Show snapshot'}
                      aria-expanded={openSnapshot === e.id}
                    >
                      <AuthedImage className="snapshot-thumb" path={api.snapshotPath(e.snapshot)} />
                    </button>
                  )}
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
