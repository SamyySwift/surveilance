import { useState } from 'react';
import { api } from '../api';

export default function EnrollModal({ availableZones = [], badgeEnabled, onClose, onEnrolled }) {
  const [name, setName] = useState('');
  const [employeeId, setEmployeeId] = useState('');
  const [badgeId, setBadgeId] = useState('');
  const [selectedZones, setSelectedZones] = useState([]);
  const [files, setFiles] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    if (!name.trim() || files.length === 0) {
      setError('Enter a name and choose at least one photo.');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const result = await api.enrollStaff(
        name.trim(),
        employeeId.trim() || null,
        badgeId.trim() || null,
        selectedZones,
        files
      );
      if (result.errors?.length) {
        setError(`Enrolled ${result.enrolled} of ${files.length} photos. ${result.errors.join(' ')}`);
      }
      onEnrolled();
    } catch (err) {
      setError(err.errors ? `${err.message}: ${err.errors.join(' ')}` : err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Enroll staff</h2>
          <button className="btn-icon" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <form onSubmit={submit} className="modal-body">
          <label className="field">
            <span>Full name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Jane Doe" required />
          </label>

          <label className="field">
            <span>Employee ID (optional)</span>
            <input
              value={employeeId}
              onChange={(e) => setEmployeeId(e.target.value)}
              placeholder="E1023"
            />
          </label>

          {badgeEnabled && (
            <label className="field">
              <span>Badge ID (optional)</span>
              <input
                value={badgeId}
                onChange={(e) => setBadgeId(e.target.value)}
                placeholder="40123"
              />
              <span className="field-hint">The number the door's badge reader sends. Needed for badge + face entry.</span>
            </label>
          )}

          <div className="field">
            <span id="enroll-zones-label">Access zones</span>
            <div className="zone-checkboxes" role="group" aria-labelledby="enroll-zones-label">
              {availableZones.map(zone => (
                <label key={zone} className="check">
                  <input
                    type="checkbox"
                    checked={selectedZones.includes(zone)}
                    onChange={(e) => {
                      if (e.target.checked) setSelectedZones([...selectedZones, zone]);
                      else setSelectedZones(selectedZones.filter(z => z !== zone));
                    }}
                  />
                  {zone}
                </label>
              ))}
            </div>
            {availableZones.length === 0 && <span className="field-hint">No zones configured.</span>}
          </div>

          <label className="field">
            <span>Photos</span>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => setFiles(Array.from(e.target.files))}
            />
            <span className="field-hint">
              2–5 clear, front-facing photos of only this person. Each is checked for sharpness, lighting
              and angle. Capturing at the door camera afterwards ("Capture" in the staff list) usually matches better.
            </span>
          </label>

          {error && <p className="form-error">{error}</p>}

          <div className="modal-actions">
            <button type="button" className="btn btn-ghost" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? 'Enrolling…' : 'Enroll'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
