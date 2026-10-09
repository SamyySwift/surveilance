import { useState } from 'react';
import { api } from '../api';
import ConfirmDialog from './ConfirmDialog';

export default function StaffPanel({ staff, zones, badgeEnabled, onRefresh, onEnroll, onManagePhotos }) {
  const activeCount = staff.filter((s) => s.active).length;
  const [query, setQuery] = useState('');
  // Which inline editor is open, for whom: { id, kind: 'zones' | 'badge' | 'camera' | 'more' }
  const [editing, setEditing] = useState(null);
  const [zoneSelection, setZoneSelection] = useState([]);
  const [badgeDraft, setBadgeDraft] = useState('');
  const [cameraZone, setCameraZone] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null); // { id, text, error }
  const [removing, setRemoving] = useState(null); // member awaiting confirmation

  const q = query.trim().toLowerCase();
  const filtered = q
    ? staff.filter((s) =>
        [s.name, s.employeeId, badgeEnabled ? s.badgeId : null, ...(s.zones || [])]
          .filter(Boolean)
          .some((v) => String(v).toLowerCase().includes(q))
      )
    : staff;

  const toggle = (member, kind) => {
    setMessage(null);
    if (editing?.id === member.id && editing.kind === kind) {
      setEditing(null);
      return;
    }
    setEditing({ id: member.id, kind });
    // Only zones that have a camera: anything else can't be shown as a checkbox,
    // so it mustn't be saved back unseen.
    if (kind === 'zones') setZoneSelection((member.zones || []).filter((z) => zones.includes(z)));
    if (kind === 'badge') setBadgeDraft(member.badgeId || '');
    if (kind === 'camera') setCameraZone(zones[0] || '');
  };
  const close = () => setEditing(null);

  const run = async (member, action, success) => {
    setBusy(true);
    try {
      const result = await action();
      close();
      if (success) setMessage({ id: member.id, text: success(result), error: false });
      onRefresh();
    } catch (err) {
      const detail = err.errors?.length ? `${err.message}: ${err.errors.join(' ')}` : err.message;
      setMessage({ id: member.id, text: detail, error: true });
    } finally {
      setBusy(false);
    }
  };

  const toggleActive = (member) =>
    run(member, () => api.updateStaff(member.id, { active: !member.active }), () =>
      member.active ? `${member.name} is suspended and will no longer match.` : `${member.name} is active again.`
    );

  const saveZones = (member) => run(member, () => api.updateStaff(member.id, { zones: zoneSelection }));

  const saveBadge = (member) =>
    run(member, () => api.updateStaff(member.id, { badgeId: badgeDraft.trim() || null }));

  const captureFromCamera = (member) =>
    run(
      member,
      () => api.enrollFromCamera(member.id, cameraZone),
      (r) =>
        `Added ${r.enrolled} photo${r.enrolled === 1 ? '' : 's'} from the ${cameraZone} camera` +
        (r.errors?.length ? `. Skipped: ${r.errors.join(' ')}` : '.')
    );

  const removeStaff = (member) => {
    setRemoving(null);
    run(member, () => api.deleteStaff(member.id));
  };

  return (
    <div className="panel staff-panel">
      <div className="panel-header">
        <h2>Staff</h2>
        <span className="panel-header-meta">
          {activeCount} active{staff.length !== activeCount ? ` · ${staff.length - activeCount} suspended` : ''}
        </span>
      </div>

      <button className="btn btn-primary enroll-btn" onClick={onEnroll}>
        Enroll new staff
      </button>

      {staff.length > 0 && (
        <input
          className="search-input"
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={`Search ${staff.length} people`}
          aria-label="Search staff by name, ID or zone"
        />
      )}

      {staff.length === 0 ? (
        <p className="empty-state">No one's enrolled yet. Add your first staff member to start recognizing faces.</p>
      ) : filtered.length === 0 ? (
        <p className="empty-state">Nobody matches “{query}”.</p>
      ) : (
        <ul className="staff-list">
          {filtered.map((member) => {
            const isEditing = (kind) => editing?.id === member.id && editing.kind === kind;
            return (
              <li key={member.id} className={`staff-row ${member.active ? '' : 'staff-row--suspended'}`}>
                <div className="staff-row-main">
                  <span className={`status-dot ${member.active ? 'status-dot--ok' : 'status-dot--muted'}`} />
                  <div className="staff-info">
                    <div className="staff-name">
                      <span className="staff-name-text">{member.name}</span>
                      {!member.active && <span className="staff-tag">Suspended</span>}
                    </div>
                    <div className="staff-meta">
                      {member.employeeId || 'No ID'}
                      {badgeEnabled && ` · badge ${member.badgeId || 'none'}`} · {member.numEmbeddings} photo
                      {member.numEmbeddings === 1 ? '' : 's'}
                    </div>
                    <div className="staff-zones">
                      {(member.zones || []).map((z) => (
                        <span key={z} className="zone-chip">{z}</span>
                      ))}
                      {(!member.zones || member.zones.length === 0) && <span className="zone-none">No access</span>}
                    </div>
                  </div>
                </div>

                <div className="staff-row-actions">
                  <button className="btn btn-ghost btn-small" aria-expanded={isEditing('zones')} onClick={() => toggle(member, 'zones')}>
                    Zones
                  </button>
                  <button className="btn btn-ghost btn-small" onClick={() => onManagePhotos(member)}>
                    Photos
                  </button>
                  <button
                    className="btn btn-ghost btn-small"
                    aria-expanded={isEditing('camera')}
                    onClick={() => toggle(member, 'camera')}
                    title="Capture enrollment photos at a door camera"
                  >
                    Capture
                  </button>
                  <button className="btn btn-ghost btn-small" aria-expanded={isEditing('more')} onClick={() => toggle(member, 'more')}>
                    More
                  </button>
                </div>

                {isEditing('zones') && (
                  <div className="inline-editor">
                    <span className="inline-hint">Where {member.name} may be:</span>
                    {zones.map((z) => (
                      <label key={z} className="check">
                        <input
                          type="checkbox"
                          checked={zoneSelection.includes(z)}
                          onChange={(e) =>
                            setZoneSelection(
                              e.target.checked ? [...zoneSelection, z] : zoneSelection.filter((x) => x !== z)
                            )
                          }
                        />
                        {z}
                      </label>
                    ))}
                    <div className="inline-buttons">
                      <button className="btn btn-primary btn-small" disabled={busy} onClick={() => saveZones(member)}>Save</button>
                      <button className="btn btn-ghost btn-small" onClick={close}>Cancel</button>
                    </div>
                  </div>
                )}

                {isEditing('camera') && (
                  <div className="inline-editor">
                    <span className="inline-hint">
                      Have {member.name} stand alone at the camera, facing it, for about 6 seconds.
                    </span>
                    <select
                      className="inline-input"
                      value={cameraZone}
                      onChange={(e) => setCameraZone(e.target.value)}
                      aria-label="Camera"
                    >
                      {zones.map((z) => <option key={z} value={z}>{z}</option>)}
                    </select>
                    <div className="inline-buttons">
                      <button className="btn btn-primary btn-small" disabled={busy || !cameraZone} onClick={() => captureFromCamera(member)}>
                        {busy ? 'Capturing…' : 'Capture'}
                      </button>
                      <button className="btn btn-ghost btn-small" disabled={busy} onClick={close}>Cancel</button>
                    </div>
                  </div>
                )}

                {isEditing('badge') && (
                  <form className="inline-editor" onSubmit={(e) => { e.preventDefault(); saveBadge(member); }}>
                    <input
                      className="inline-input"
                      value={badgeDraft}
                      onChange={(e) => setBadgeDraft(e.target.value)}
                      placeholder="Badge ID (empty to remove)"
                      aria-label={`Badge ID for ${member.name}`}
                      autoFocus
                    />
                    <div className="inline-buttons">
                      <button type="submit" className="btn btn-primary btn-small" disabled={busy}>Save</button>
                      <button type="button" className="btn btn-ghost btn-small" onClick={close}>Cancel</button>
                    </div>
                  </form>
                )}

                {isEditing('more') && (
                  <div className="inline-editor">
                    {badgeEnabled && (
                      <button className="btn btn-ghost btn-small" onClick={() => toggle(member, 'badge')}>Badge</button>
                    )}
                    <button className="btn btn-ghost btn-small" disabled={busy} onClick={() => toggleActive(member)}>
                      {member.active ? 'Suspend' : 'Reinstate'}
                    </button>
                    <button className="btn btn-ghost btn-small btn-danger" disabled={busy} onClick={() => setRemoving(member)}>
                      Remove
                    </button>
                  </div>
                )}

                {message?.id === member.id && (
                  <p className={message.error ? 'form-error' : 'form-note'} role="status">{message.text}</p>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {removing && (
        <ConfirmDialog
          title={`Remove ${removing.name}?`}
          confirmLabel="Remove"
          danger
          onCancel={() => setRemoving(null)}
          onConfirm={() => removeStaff(removing)}
        >
          <p className="modal-text">
            This deletes {removing.name} and all their enrolled photos. They'll no longer be recognized. It can't be
            undone.
          </p>
        </ConfirmDialog>
      )}
    </div>
  );
}
