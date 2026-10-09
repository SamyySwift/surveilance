import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { formatTime } from '../format';
import AuthedImage from './AuthedImage';
import ConfirmDialog from './ConfirmDialog';
import EmailReportModal from './EmailReportModal';

const SEVERITY = {
  high: { icon: '▲', label: 'High' },
  medium: { icon: '●', label: 'Medium' },
  info: { icon: 'i', label: 'Info' },
};
const PAGE = 8;

export default function AlarmLog({ alarms, onAcknowledge, onClear, emailSettings }) {
  const [filter, setFilter] = useState('open');
  const [limit, setLimit] = useState(PAGE);
  const [flashId, setFlashId] = useState(null);
  const [openSnapshot, setOpenSnapshot] = useState(null);
  // Open dialog: the alarm being acknowledged, or { upToId } for clearing.
  const [acking, setAcking] = useState(null);
  const [clearing, setClearing] = useState(null);
  const [emailing, setEmailing] = useState(false);
  const newestRef = useRef(alarms[0]?.id);

  useEffect(() => {
    const newest = alarms[0]?.id;
    if (newest !== undefined && newestRef.current !== undefined && newest > newestRef.current) {
      setFlashId(newest);
      const timer = setTimeout(() => setFlashId(null), 1400);
      newestRef.current = newest;
      return () => clearTimeout(timer);
    }
    newestRef.current = newest;
  }, [alarms]);

  const open = alarms.filter((a) => !a.acknowledgedAt);
  const shown = filter === 'open' ? open : alarms;
  const visible = shown.slice(0, limit);

  const choose = (f) => {
    setFilter(f);
    setLimit(PAGE);
  };

  // Clearing stops at the newest alarm on screen when the dialog opened, so
  // one that arrives while it's open is kept.
  const openClear = () => setClearing({ upToId: Math.max(...alarms.map((a) => a.id)) });
  const openToClear = clearing ? open.filter((a) => a.id <= clearing.upToId).length : 0;

  return (
    <div className="panel alarm-panel">
      <div className="panel-header">
        <h2>Alarms</h2>
        <div className="panel-header-actions">
          <div className="segmented" role="group" aria-label="Which alarms to show">
            <button aria-pressed={filter === 'open'} onClick={() => choose('open')}>
              Open {open.length}
            </button>
            <button aria-pressed={filter === 'all'} onClick={() => choose('all')}>
              All {alarms.length}
            </button>
          </div>
          <button className="btn btn-ghost btn-small" onClick={() => setEmailing(true)}>
            Email
          </button>
          {alarms.length > 0 && (
            <button className="btn btn-ghost btn-small btn-danger" onClick={openClear}>
              Clear
            </button>
          )}
        </div>
      </div>

      {shown.length === 0 ? (
        <p className="empty-state">
          {filter === 'open' && alarms.length > 0
            ? 'No open alarms. Everything has been acknowledged.'
            : 'No alarms yet. Unknown people, failed liveness checks and camera problems show up here.'}
        </p>
      ) : (
        <ul className="alarm-list">
          {visible.map((a) => {
            const sev = SEVERITY[a.severity] || SEVERITY.medium;
            return (
              <li
                key={a.id}
                className={`alarm-row alarm-row--${a.severity} ${a.acknowledgedAt ? 'alarm-row--acked' : ''} ${
                  a.id === flashId ? 'alarm-row--flash' : ''
                }`}
              >
                <span className={`severity severity--${a.severity}`} title={`${sev.label} severity`}>
                  <span aria-hidden="true">{sev.icon}</span> {sev.label}
                </span>
                <div className="alarm-body">
                  <div className="alarm-message">{a.message}</div>
                  <div className="alarm-meta">
                    {formatTime(a.time)}
                    {a.zone && ` · ${a.zone}`}
                    {a.acknowledgedAt &&
                      ` · acknowledged ${formatTime(a.acknowledgedAt)}${a.acknowledgedNote ? `: ${a.acknowledgedNote}` : ''}`}
                  </div>
                  {openSnapshot === a.id && a.snapshot && (
                    <AuthedImage className="snapshot-large" path={api.snapshotPath(a.snapshot)} alt="Alarm snapshot" />
                  )}
                </div>
                <div className="row-actions">
                  <span className="thumb-slot">
                    {a.snapshot && (
                      <button
                        className="snapshot-thumb-btn"
                        onClick={() => setOpenSnapshot(openSnapshot === a.id ? null : a.id)}
                        aria-label={openSnapshot === a.id ? 'Hide snapshot' : 'Show snapshot'}
                        aria-expanded={openSnapshot === a.id}
                      >
                        <AuthedImage className="snapshot-thumb" path={api.snapshotPath(a.snapshot)} />
                      </button>
                    )}
                  </span>
                  {a.acknowledgedAt ? (
                    <span className="acked-label">✓ Done</span>
                  ) : (
                    <button className="btn btn-ghost btn-small" onClick={() => setAcking(a)}>
                      Acknowledge
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {shown.length > limit && (
        <button className="btn btn-ghost btn-small show-more" onClick={() => setLimit(limit + PAGE)}>
          {shown.length - limit <= PAGE
            ? `Show ${shown.length - limit} more`
            : `Show ${PAGE} more (${shown.length - limit} not shown)`}
        </button>
      )}

      {acking && (
        <ConfirmDialog
          title="Acknowledge alarm"
          confirmLabel="Acknowledge"
          busyLabel="Saving…"
          noteLabel="Note (optional)"
          notePlaceholder="What did you find?"
          onCancel={() => setAcking(null)}
          onConfirm={async (note) => {
            await onAcknowledge(acking.id, note);
            setAcking(null);
          }}
        >
          <AlarmSummary alarm={acking} />
        </ConfirmDialog>
      )}

      {clearing && (
        <ConfirmDialog
          title="Clear alarm log"
          confirmLabel="Delete alarms"
          busyLabel="Deleting…"
          danger
          onCancel={() => setClearing(null)}
          onConfirm={async () => {
            await onClear(clearing.upToId);
            setClearing(null);
          }}
        >
          <p className="modal-text">
            This permanently deletes every alarm in the log and its snapshot. It can't be undone.
          </p>
          {openToClear > 0 && (
            <p className="modal-warning">
              {openToClear === 1
                ? "1 alarm is still open and hasn't been acknowledged."
                : `${openToClear} alarms are still open and haven't been acknowledged.`}
            </p>
          )}
        </ConfirmDialog>
      )}

      {emailing && (
        <EmailReportModal settings={emailSettings} openCount={open.length} onClose={() => setEmailing(false)} />
      )}
    </div>
  );
}

// The alarm a dialog is about, as it appears in the log.
function AlarmSummary({ alarm }) {
  const sev = SEVERITY[alarm.severity] || SEVERITY.medium;
  return (
    <div className={`dialog-subject dialog-subject--${alarm.severity}`}>
      <span className={`severity severity--${alarm.severity}`}>
        <span aria-hidden="true">{sev.icon}</span> {sev.label}
      </span>
      <div className="alarm-message">{alarm.message}</div>
      <div className="alarm-meta">
        {formatTime(alarm.time)}
        {alarm.zone && ` · ${alarm.zone}`}
      </div>
    </div>
  );
}
